#pragma once

#include <type_traits>
#include <complex>

#include "util.h"

template <typename T,
          typename = std::enable_if<my_is_floating_point_v<T>, T>>
struct body
{
	body(T x, T y, T mass) : x(x), y(y), mass(mass)
	{
	}

	T x;
	T y;
	T mass;

	std::complex<T> pos() { return std::complex<T>(x, y); }
};
