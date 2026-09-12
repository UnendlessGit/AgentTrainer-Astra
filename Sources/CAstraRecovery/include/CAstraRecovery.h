#ifndef ASTRA_RECOVERY_H
#define ASTRA_RECOVERY_H
#include <stdint.h>
#include <stdbool.h>
#include <sys/types.h>

#define ASTRA_RECOVERY_BYTES 4096
enum astra_recovery_phase { ASTRA_PREPARING=1, ASTRA_ARMED=2, ASTRA_STOPPING=3, ASTRA_RECOVERING=4, ASTRA_SETTLED=5 };
typedef struct {
    uint8_t run[16], ledger[16];
    uint64_t capabilities[3], possible[3];
    uint64_t guardian_heartbeat, completed_nanos, guardian_identity;
    uint32_t phase, in_flight, executor_pid, guardian_pid, guardian_ready, ever_armed, performer, error_code;
} astra_recovery_snapshot;
bool astra_recovery_initialize(void *, const uint8_t *, const uint8_t *, const uint64_t *);
bool astra_recovery_read(const void *, astra_recovery_snapshot *);
bool astra_recovery_executor(void *, uint32_t);
bool astra_recovery_guardian(void *, uint32_t, uint64_t);
void astra_recovery_heartbeat(void *, uint64_t);
bool astra_recovery_arm(void *);
bool astra_recovery_guardian_fresh(const void *, uint64_t, uint64_t);
void astra_recovery_stop(void *);
bool astra_recovery_begin_post(void *, uint32_t, uint32_t, bool);
void astra_recovery_end_post(void *, uint32_t, uint32_t, bool);
bool astra_recovery_local_settle(void *, uint64_t);
bool astra_recovery_claim_after_exit(void *);
bool astra_recovery_release(void *, uint32_t, uint32_t);
bool astra_recovery_guardian_settle(void *, uint64_t);
void astra_recovery_error(void *, uint32_t);
int astra_guardian_spawn(const char *, int, int, pid_t, pid_t *);
uint64_t astra_process_identity(pid_t);
#endif
