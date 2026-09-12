#include "CAstraRecovery.h"
#include <stdatomic.h>
#include <string.h>
#include <spawn.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <libproc.h>
#include <sys/proc.h>

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint8_t run[16], ledger[16];
    uint64_t capabilities[3];
    _Alignas(64) _Atomic uint64_t possible[3];
    _Atomic uint64_t guardian_heartbeat, completed_nanos, guardian_identity;
    _Atomic uint32_t phase, in_flight, executor_pid, guardian_pid, guardian_ready, ever_armed, performer, error_code;
} recovery_page;
_Static_assert(sizeof(recovery_page) <= ASTRA_RECOVERY_BYTES, "Recovery page exceeds fixed mapping");
static const uint64_t magic = UINT64_C(0x4153545241524331);
static bool valid(const recovery_page *p) {
    return p && p->magic == magic && p->version == 1 &&
        atomic_is_lock_free(&p->possible[0]) && atomic_is_lock_free(&p->phase);
}
static bool control_bit(uint32_t kind, uint32_t code, uint32_t *word, uint64_t *bit) {
    if (kind == 1 && code < 128) { *word=code/64; *bit=UINT64_C(1)<<(code%64); return true; }
    if (kind == 2 && code < 32) { *word=2; *bit=UINT64_C(1)<<code; return true; }
    return false;
}
bool astra_recovery_initialize(void *address, const uint8_t *run, const uint8_t *ledger, const uint64_t *caps) {
    if (!address || ((uintptr_t)address % 64) != 0) return false;
    memset(address,0,ASTRA_RECOVERY_BYTES);
    recovery_page *p=address; p->version=1;
    memcpy(p->run,run,16); memcpy(p->ledger,ledger,16); memcpy(p->capabilities,caps,24);
    for (int i=0;i<3;i++) atomic_init(&p->possible[i],0);
    atomic_init(&p->guardian_heartbeat,0); atomic_init(&p->completed_nanos,0);atomic_init(&p->guardian_identity,0);
    atomic_init(&p->phase,ASTRA_PREPARING); atomic_init(&p->in_flight,0);
    atomic_init(&p->executor_pid,0); atomic_init(&p->guardian_pid,0); atomic_init(&p->guardian_ready,0);
    atomic_init(&p->performer,0); atomic_init(&p->error_code,0);atomic_init(&p->ever_armed,0);
    p->magic=magic; atomic_thread_fence(memory_order_seq_cst); return valid(p);
}
bool astra_recovery_read(const void *address, astra_recovery_snapshot *out) {
    const recovery_page *p=address; if (!valid(p) || !out) return false;
    memcpy(out->run,p->run,16); memcpy(out->ledger,p->ledger,16); memcpy(out->capabilities,p->capabilities,24);
    out->phase=atomic_load(&p->phase);
    for(int i=0;i<3;i++) out->possible[i]=atomic_load(&p->possible[i]);
    out->guardian_heartbeat=atomic_load(&p->guardian_heartbeat); out->completed_nanos=atomic_load(&p->completed_nanos);
    out->guardian_identity=atomic_load(&p->guardian_identity);out->ever_armed=atomic_load(&p->ever_armed);
    out->in_flight=atomic_load(&p->in_flight); out->executor_pid=atomic_load(&p->executor_pid);
    out->guardian_pid=atomic_load(&p->guardian_pid); out->guardian_ready=atomic_load(&p->guardian_ready);
    out->performer=atomic_load(&p->performer); out->error_code=atomic_load(&p->error_code);
    return out->phase == atomic_load(&p->phase) && out->phase>=ASTRA_PREPARING && out->phase<=ASTRA_SETTLED;
}
bool astra_recovery_executor(void *address,uint32_t pid) {
    recovery_page *p=address; if(!valid(p) || atomic_load(&p->phase)!=ASTRA_PREPARING || !pid)return false;
    uint32_t empty=0; return atomic_compare_exchange_strong(&p->executor_pid,&empty,pid);
}
bool astra_recovery_guardian(void *address,uint32_t pid,uint64_t now) {
    recovery_page *p=address;if(!valid(p)||!pid||atomic_load(&p->phase)!=ASTRA_PREPARING)return false;
    uint32_t empty=0; if(!atomic_compare_exchange_strong(&p->guardian_pid,&empty,pid))return false;
    uint64_t identity=astra_process_identity(pid);if(!identity)return false;
    atomic_store(&p->guardian_identity,identity);atomic_store(&p->guardian_heartbeat,now); atomic_store(&p->guardian_ready,1);return true;
}
void astra_recovery_heartbeat(void *address,uint64_t now) { recovery_page*p=address; if(valid(p))atomic_store(&p->guardian_heartbeat,now); }
bool astra_recovery_arm(void *address) {
    recovery_page*p=address;if(!valid(p)||!atomic_load(&p->guardian_ready))return false;
    atomic_store(&p->ever_armed,1);
    uint32_t expected=ASTRA_PREPARING;return atomic_compare_exchange_strong(&p->phase,&expected,ASTRA_ARMED);
}
bool astra_recovery_guardian_fresh(const void *address,uint64_t now,uint64_t limit) {
    const recovery_page*p=address;if(!valid(p)||atomic_load(&p->phase)!=ASTRA_ARMED||!atomic_load(&p->guardian_ready))return false;
    uint64_t observed=atomic_load(&p->guardian_heartbeat);return now>=observed && now-observed<limit;
}
void astra_recovery_stop(void *address) {
    recovery_page*p=address;if(!valid(p))return;
    uint32_t value=atomic_load(&p->phase);
    while(value==ASTRA_PREPARING||value==ASTRA_ARMED)if(atomic_compare_exchange_weak(&p->phase,&value,ASTRA_STOPPING))break;
}
bool astra_recovery_begin_post(void *address,uint32_t kind,uint32_t code,bool down) {
    recovery_page*p=address;if(!valid(p)||atomic_load(&p->phase)!=ASTRA_ARMED)return false;
    uint32_t word=0;uint64_t bit=0;
    if((down && !kind) || (kind && (!control_bit(kind,code,&word,&bit)||!(p->capabilities[word]&bit))))return false;
    atomic_fetch_add(&p->in_flight,1);
    if(atomic_load(&p->phase)!=ASTRA_ARMED){atomic_fetch_sub(&p->in_flight,1);return false;}
    if(down)atomic_fetch_or(&p->possible[word],bit);
    return true;
}
void astra_recovery_end_post(void *address,uint32_t kind,uint32_t code,bool released) {
    recovery_page*p=address;if(!valid(p))return;
    uint32_t word;uint64_t bit;
    if(released && atomic_load(&p->phase)==ASTRA_ARMED && control_bit(kind,code,&word,&bit))atomic_fetch_and(&p->possible[word],~bit);
    atomic_fetch_sub(&p->in_flight,1);
}
bool astra_recovery_local_settle(void *address,uint64_t now) {
    recovery_page*p=address;if(!valid(p)||atomic_load(&p->phase)!=ASTRA_STOPPING||atomic_load(&p->in_flight))return false;
    // Caller proves its local possibly-posted ledger and every driver/cleanup
    // have settled. Bits retained through provisional cleanup can now clear.
    for(int i=0;i<3;i++)atomic_store(&p->possible[i],0);
    atomic_store(&p->performer,1);atomic_store(&p->completed_nanos,now);atomic_store(&p->error_code,0);
    atomic_store(&p->phase,ASTRA_SETTLED);return true;
}
bool astra_recovery_claim_after_exit(void *address) {
    recovery_page*p=address;if(!valid(p))return false;
    uint32_t value=atomic_load(&p->phase);
    while(value>=ASTRA_PREPARING && value<=ASTRA_STOPPING) {
        if(atomic_compare_exchange_weak(&p->phase,&value,ASTRA_RECOVERING)){
            atomic_store(&p->in_flight,0);return true;
        }
    }
    return value==ASTRA_RECOVERING;
}
bool astra_recovery_release(void *address,uint32_t kind,uint32_t code) {
    recovery_page*p=address;uint32_t word;uint64_t bit;
    if(!valid(p)||atomic_load(&p->phase)!=ASTRA_RECOVERING||!control_bit(kind,code,&word,&bit))return false;
    atomic_fetch_and(&p->possible[word],~bit);return true;
}
bool astra_recovery_guardian_settle(void *address,uint64_t now) {
    recovery_page*p=address;if(!valid(p)||atomic_load(&p->phase)!=ASTRA_RECOVERING||atomic_load(&p->in_flight))return false;
    for(int i=0;i<3;i++)if(atomic_load(&p->possible[i]))return false;
    atomic_store(&p->performer,2);atomic_store(&p->completed_nanos,now);atomic_store(&p->error_code,0);
    atomic_store(&p->phase,ASTRA_SETTLED);return true;
}
void astra_recovery_error(void *address,uint32_t code) { recovery_page*p=address;if(valid(p))atomic_store(&p->error_code,code); }

extern char **environ;
uint64_t astra_process_identity(pid_t pid) {
    struct proc_bsdinfo info;
    int count=proc_pidinfo(pid,PROC_PIDTBSDINFO,0,&info,sizeof(info));
    if(count!=sizeof(info) || info.pbi_status==SZOMB)return 0;
    return (uint64_t)info.pbi_start_tvsec*UINT64_C(1000000)+(uint64_t)info.pbi_start_tvusec;
}
int astra_guardian_spawn(const char *path,int lease,int ledger,pid_t owner,pid_t *pid) {
    int lease_copy=fcntl(lease,F_DUPFD_CLOEXEC,64);if(lease_copy<0)return errno;
    int ledger_copy=fcntl(ledger,F_DUPFD_CLOEXEC,64);if(ledger_copy<0){int e=errno;close(lease_copy);return e;}
    posix_spawn_file_actions_t actions;posix_spawnattr_t attributes;
    int result=posix_spawn_file_actions_init(&actions);if(result)goto done;
    result=posix_spawnattr_init(&attributes);if(result){posix_spawn_file_actions_destroy(&actions);goto done;}
    if(!(result=posix_spawnattr_setflags(&attributes,POSIX_SPAWN_CLOEXEC_DEFAULT))) {
        for(int fd=0;fd<3 && !result;fd++)result=posix_spawn_file_actions_addopen(&actions,fd,"/dev/null",O_RDWR,0);
        if(!result)result=posix_spawn_file_actions_adddup2(&actions,lease_copy,3);
        if(!result)result=posix_spawn_file_actions_adddup2(&actions,ledger_copy,4);
        char owner_text[32];snprintf(owner_text,sizeof(owner_text),"%d",owner);
        char *argv[]={(char*)path,"--guardian",owner_text,NULL};
        if(!result)result=posix_spawn(pid,path,&actions,&attributes,argv,environ);
    }
    posix_spawn_file_actions_destroy(&actions);posix_spawnattr_destroy(&attributes);
done:close(lease_copy);close(ledger_copy);return result;
}
