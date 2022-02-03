#pragma once
#include <array>

constexpr unsigned k = 2;
using point_t = std::array<float, 2>;

// A structure to represent node of kd tree
struct node
{
	node() = default;
	node(float x, float y);
	point_t point;

	node* left;
	node* right;
};

class kd_tree
{
public:
	kd_tree();

	void insert_point(const point_t& point);

	float find_min(unsigned dim) const;

	point_t find_nearest_neighbor(const point_t& point);


private:
	node* root_;
};
