#include <array>
#include <cassert>
#include <chrono>
#include <iostream>
#include <vector>

#include "util.h"
#include "accumulator.h"
#include "bh_tree.h"
#include "body.h"

void run_bh_cuda(const std::vector<std::shared_ptr<body<float>>>& bodies,
                 std::pair<float, float>* us,
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

		std::pair<float, float> result{};
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

int main()
{
	constexpr int num_bodies = 1024 * 10;
	assert(num_bodies >= 1024);

	// Inputs
	std::vector<std::shared_ptr<body<float>>> bodies;
	bodies.reserve(num_bodies);

	for (int i = 0; i < num_bodies; ++i)
	{
		bodies.push_back(std::make_unique<body<float>>(my_rand(), my_rand(), my_rand() * 1.5f));
	}

	// Outputs
	const auto us = static_cast<std::pair<float, float>*>(calloc(num_bodies, sizeof(std::pair<float, float>)));

	// Run
	run_bh_cuda(bodies, us, 0.75f);

	// Print result, some sneaky peek + RMSE 
	if (us != nullptr)
	{
		for (int i = 0; i < 10; ++i)
		{
			std::cout << '(' << us[i].first << ", " << us[i].second << ')' << std::endl;
		}
	}

	std::cout << "RMSE: " << compute_rmse(bodies, us, 1024) << std::endl;

	return EXIT_SUCCESS;
}
