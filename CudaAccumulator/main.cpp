#include <array>
#include <chrono>
#include <iostream>

#include "util.h"
#include "accumulator.h"
#include "bh_tree.h"
#include "body.h"

void run_naive_cuda(const std::vector<std::shared_ptr<body<float>>>& bodies,
                    std::pair<float, float>* us)
{
	const size_t num_bodies = bodies.size();

	accumulator_handle* acc = get_accumulator();
	for (size_t i = 0; i < num_bodies; ++i)
	{
		accumulator_set_constants_and_result_address(bodies[i]->x, bodies[i]->y, &us[i].first, acc);

		for (size_t j = 0; j < num_bodies; ++j)
		{
			accumulator_accumulate(bodies[j]->x, bodies[j]->y, bodies[j]->mass, acc);
		}

		accumulator_finish(acc);
	}

	release_accumulator(acc);
}

void run_bh_cuda(const std::vector<std::shared_ptr<body<float>>>& bodies,
                 std::pair<float, float>* us)
{
	std::cout << "BH: Building the quadtree." << std::endl;

	TIME_THIS_SEGMENT(

		auto qt = barnes_hut::quadtree();

		for (const auto& body : bodies)
		{
		qt.allocate_node_for_particle(body);
		}

		qt.compute_center_of_mass();

	)

	std::cout << "BH: Start Traversing the tree..." << std::endl;

	accumulator_handle* acc = get_accumulator();

	const size_t num_bodies = bodies.size();
	for (size_t i = 0; i < num_bodies; ++i)
	{
		const auto pos = bodies[i]->pos();

		std::pair<float, float> result{};
		accumulator_set_constants_and_result_address(pos.real(), pos.imag(), &result.first, acc);

		qt.compute_force_accumulator(acc, 1.0f);

		us[i].first += result.first;
		us[i].second += result.second;
	}

	release_accumulator(acc);

	std::cout << "BH: Done! " << std::endl;
}

int main()
{
	constexpr int num_bodies = 1024 * 100;

	// Inputs
	std::vector<std::shared_ptr<body<float>>> bodies;

	bodies.reserve(num_bodies);
	for (int i = 0; i < num_bodies; ++i)
	{
		bodies.push_back(std::make_unique<body<float>>(my_rand(), my_rand(), my_rand() * 1.5f));
	}

	// Outputs
	const auto us = static_cast<std::pair<float, float>*>(malloc(num_bodies * sizeof(std::pair<float, float>)));
	if (us != nullptr)
	{
		for (int i = 0; i < num_bodies; ++i)
		{
			us[i].first = 0.0f;
			us[i].second = 0.0f;
		}
	}

	//run_naive_cuda(bodies, us.data());
	run_bh_cuda(bodies, us);

	// Print result
	if (us != nullptr)
	{
		for (int i = 0; i < 10; ++i)
		{
			std::cout << '(' << us[i].first << ", " << us[i].second << ')' << std::endl;
		}
	}

	print_ground_truth(bodies);

	return EXIT_SUCCESS;
}
