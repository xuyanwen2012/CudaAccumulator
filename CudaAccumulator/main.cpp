#include <array>
#include <iostream>
#include <random>

#include "accumulator.h"
#include "bh_tree.h"
#include "body.h"

float my_rand(const float f_min = 0.0, const float f_max = 1.0)
{
	const float f = static_cast<float>(rand()) / RAND_MAX;
	return f_min + f * (f_max - f_min);
}

void print_ground_truth(const std::vector<std::shared_ptr<body<float>>>& bodies)
{
	constexpr unsigned n_to_print = 10;
	std::array<std::pair<float, float>, n_to_print> us{};

	for (unsigned i = 0; i < n_to_print; ++i)
	{
		us[i] = {0.0f, 0.0f};
		for (unsigned j = 0; j < bodies.size(); ++j)
		{
			const float dx = bodies[i]->x - bodies[j]->x;
			const float dy = bodies[i]->y - bodies[j]->y;

			const float dist_sqr = dx * dx + dy * dy + 1e-9f;
			const float inv_dist = 1.0f / sqrtf(dist_sqr);
			const float inv_dist3 = inv_dist * inv_dist * inv_dist;
			const float with_mass = inv_dist3 * bodies[j]->mass; // z is the mass in this case

			us[i].first += dx * with_mass;
			us[i].second += dy * with_mass;
		}
	}

	std::cout << "==================" << std::endl;
	for (unsigned i = 0; i < n_to_print; ++i)
	{
		std::cout << '('
			<< us[i].first << ", "
			<< us[i].second << ')' << std::endl;
	}
}

void run_naive_cuda(const std::vector<std::shared_ptr<body<float>>>& bodies,
                    std::pair<float, float>* us,
                    const int num_bodies)
{
	accumulator_handle* acc = get_accumulator();

	for (int i = 0; i < num_bodies; ++i)
	{
		accumulator_set_constants_and_result_address(bodies[i]->x, bodies[i]->y, &us[i].first, acc);

		for (int j = 0; j < num_bodies; ++j)
		{
			accumulator_accumulate(bodies[j]->x, bodies[j]->y, bodies[j]->mass, acc);
		}
	}

	release_accumulator(acc);
}

void run_bh_cuda(const std::vector<std::shared_ptr<body<float>>>& bodies,
                 std::pair<float, float>* us,
                 const int num_bodies)
{
	std::cout << "BH: Building the quadtree." << std::endl;

	auto qt = barnes_hut::quadtree();

	for (const auto& body : bodies)
	{
		qt.allocate_node_for_particle(body);
	}

	qt.compute_center_of_mass();

	std::cout << "BH: Start Traversing the tree..." << std::endl;

	accumulator_handle* acc = get_accumulator();

	for (int i = 0; i < num_bodies; ++i)
	{
		const auto force = qt.compute_force_accumulator(acc, bodies[i]->pos(), 0.0f);
		us[i].first += force.real();
		us[i].second += force.imag();
	}

	release_accumulator(acc);

	std::cout << "BH: Done! " << std::endl;
}

int main()
{
	constexpr int num_bodies = 1234;

	// Inputs
	std::vector<std::shared_ptr<body<float>>> bodies;

	bodies.reserve(num_bodies);
	for (int i = 0; i < num_bodies; ++i)
	{
		bodies.push_back(std::make_unique<body<float>>(my_rand(), my_rand(), my_rand() * 1.5f));
	}

	// Outputs
	//float2* us = new float2[num];
	std::array<std::pair<float, float>, num_bodies> us{};

	//run_naive_cuda(bodies, us.data(), num_bodies);
	run_bh_cuda(bodies, us.data(), num_bodies);

	// Print result
	for (int i = 0; i < 10; ++i)
	{
		std::cout << '(' << us[i].first << ", " << us[i].second << ')' << std::endl;
	}

	print_ground_truth(bodies);

	return EXIT_SUCCESS;
}
