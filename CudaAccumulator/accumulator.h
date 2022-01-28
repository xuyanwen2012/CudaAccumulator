#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
using accumulator_handle = struct accumulator_handle;
#else
typedef struct accumulator_handle accumulator_handle;
#endif

accumulator_handle* get_accumulator(void);

int accumulator_set_constants_and_result_address(float x, float y, float* addr, accumulator_handle* acc);

int accumulator_accumulate(float x, float y, float mass, accumulator_handle* acc);

int accumulator_finish(accumulator_handle* acc);

int release_accumulator(accumulator_handle* acc);

#ifdef __cplusplus
}
#endif
