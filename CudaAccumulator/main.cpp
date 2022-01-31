#include <array>
#include <cassert>
#include <chrono>
#include <iostream>
#include <vector>

#include "util.h"
#include "accumulator.h"
#include "bh_tree.h"
#include "body.h"

void run_bh_cuda(const body_container& bodies,
                 pair_f* us,
                 const float theta = 1.0f)
{
	std::cout << "BH: Building the quadtree." << std::endl;

	auto qt = barnes_hut::quadtree();
	for (const auto& body : bodies)
	{
		qt.allocate_node_for_particle(body);
	}
	qt.compute_center_of_mass();

	std::cout << "BH: Start Traversing the tree..." << std::endl;
	const auto start = std::chrono::steady_clock::now();

	accumulator_handle* acc = get_accumulator();

	const size_t num_bodies = bodies.size();
	for (size_t i = 0; i < num_bodies; ++i)
	{
		const auto pos = bodies[i]->pos();

		pair_f result{};
		accumulator_set_constants_and_result_address(pos.real(), pos.imag(), &result.first, acc);

		qt.compute_force_accumulator(acc, theta);

		us[i].first += result.first;
		us[i].second += result.second;
	}

	release_accumulator(acc);

	const auto end = std::chrono::steady_clock::now();
	const std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "- elapsed time: " << elapsed_seconds.count() << 's' << std::endl;
	std::cout << "BH: Done! " << std::endl;
}

body_container init_bodies(int n)
{
	body_container bodies;
	bodies.reserve(n);

	for (int i = 0; i < n; ++i)
	{
		bodies.push_back(std::make_unique<body<float>>(my_rand(), my_rand(), my_rand() * 1.5f));
	}

	return bodies;
}

int main()
{
	constexpr int num_bodies = 1024;
	assert(num_bodies >= 1024);

	const body_container bodies = init_bodies(num_bodies);

	const auto us = make_output_array<pair_f>(num_bodies);

	// Run
	run_bh_cuda(bodies, us, 0.0f);

	// Print result, some sneaky peek + RMSE 
	for (int i = 0; i < 10; ++i)
	{
		std::cout << '(' << us[i].first << ", " << us[i].second << ')' << std::endl;
	}

	std::cout << "RMSE: " << compute_rmse(bodies, us, 1024) << std::endl;

	return EXIT_SUCCESS;
}
