#pragma once

#ifdef __cplusplus
extern "C" {
#endif

using accumulator_handle = struct accumulator_handle;

accumulator_handle* get_accumulator(void);

/**
 * \brief pass in some constant values (x, y) of particle i that will be used when doing the kernel
 * computation.
 * \param x the x position of the particle i.
 * \param y the y position of the particle i.
 * \param addr pointer to where the result (potential, u of particle i) is stored after accumulation.
 * \param acc which accumulator_handle is used?
 * \return successful or not (1 or 0)
 */
int accumulator_set_constants_and_result_address(double x, double y, double* addr, accumulator_handle* acc);

/**
 * \brief send particle j's data to the accelerator so it can do the accumulation.
 * \param x the x position of the particle j.
 * \param y the y position of the particle j.
 * \param mass the mass of the particle j.
 * \param acc which accumulator_handle is used?
 * \return successful or not (1 or 0)
 */
int accumulator_accumulate(double x, double y, double mass, accumulator_handle* acc);

int release_accumulator(const accumulator_handle* ret);

#ifdef __cplusplus
}
#endif
