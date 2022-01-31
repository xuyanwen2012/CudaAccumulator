#pragma once

#include <type_traits>

template <class Ty>
constexpr bool my_is_floating_point_v = std::_Is_any_of_v<std::remove_cv_t<Ty>, float, double, long double>;

float my_rand();
