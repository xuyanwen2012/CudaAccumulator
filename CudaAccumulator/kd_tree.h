#pragma once

constexpr unsigned k = 2;
using point_t = float[k];

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

	node* insert_point(const point_t& point);

private:
	static node* inner_insert_point(node* current, const point_t& point, unsigned depth);

	node* root_;
};
