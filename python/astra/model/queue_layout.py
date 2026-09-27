"""Schema-3 queued-control feature layout, independent of MLX allocations.

IDs authenticate the source ledger, but never appear in these neural features.
This feature order and interpolation semantics are part of model schema 3.
"""
MAXIMUM_PACKETS = 64
CATEGORICAL_NAMES = ('operation', 'key', 'button', 'surface', 'progress', 'anchor_operation', 'anchor_surface')
EMBEDDING_WIDTHS = (8, 16, 8, 8, 8, 4, 4)
PACKET_STATUS_WIDTH = 8
COMMAND_FEATURE_NAMES = (
    'offset_fraction', 'scheduled_seconds_clipped', 'scheduled_signed_log',
    'x', 'y', 'dx_scaled', 'dy_scaled', 'dx_signed_log', 'dy_signed_log',
    'anchor_present', 'anchor_offset_fraction', 'anchor_x', 'anchor_y',
    'anchor_dx_scaled', 'anchor_dy_scaled', 'anchor_dx_signed_log', 'anchor_dy_signed_log',
    'completed_fraction', 'completed_samples_log', 'has_completed', 'last_completed_offset_fraction',
    'last_completed_age_log', 'has_posted', 'last_posted_age_log', 'emitted_dx_scaled', 'emitted_dy_scaled',
)
PACKET_FEATURE_NAMES = (
    'admitted_age_seconds_clipped', 'admitted_age_log', 'start_seconds_clipped', 'start_signed_log',
    'end_seconds_clipped', 'end_signed_log', 'duration_seconds', 'command_fraction',
    'admission_changed', 'terminal_changed', 'terminal_age_log',
)
COMMAND_FEATURES = {name: index for index, name in enumerate(COMMAND_FEATURE_NAMES)}
PACKET_FEATURES = {name: index for index, name in enumerate(PACKET_FEATURE_NAMES)}
OPERATIONS = {'keyDown': 1, 'keyUp': 2, 'keyRepeat': 3, 'buttonDown': 4, 'buttonUp': 5,
              'pointerAbsolute': 6, 'pointerRelative': 7, 'scroll': 8}
PROGRESS = {'pending': 1, 'partial': 2, 'posted': 3, 'noOp': 4, 'cancelled': 5, 'failed': 6}
PACKET_STATUS = {'executed': 1, 'cancelled': 2, 'late': 3}  # Zero is an outstanding packet.


def category_sizes(maximum_surfaces):
    # Key/button/surface zero means not applicable, with semantic values + 1.
    return (9, 129, 33, maximum_surfaces + 1, 7, 9, maximum_surfaces + 1)


def parameter_count(*, maximum_surfaces, command_width, packet_width, recurrent_width):
    linear = lambda input_width, output_width: (input_width + 1) * output_width
    gru = lambda width: 6 * width * width + 4 * width
    categories = sum(size * width for size, width in zip(category_sizes(maximum_surfaces), EMBEDDING_WIDTHS))
    command_input = sum(EMBEDDING_WIDTHS) + len(COMMAND_FEATURE_NAMES)
    packet_input = command_width + PACKET_STATUS_WIDTH + len(PACKET_FEATURE_NAMES)
    return (categories + linear(command_input, command_width) + 2 * command_width + gru(command_width)
            + 4 * PACKET_STATUS_WIDTH + linear(packet_input, packet_width) + 2 * packet_width + gru(packet_width)
            + packet_width * recurrent_width)  # Bias-free zero-initialized residual output.
