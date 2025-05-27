#ifndef _BASE_FUNCTION_HPP_
#define _BASE_FUNCTION_HPP_
#include <math.h>

#include "base_logic.hpp"
#include "mat.hpp"

template<typename func_t>
auto derivative(func_t&& f, const decltype(f(0))& v)
{
	constexpr double SMALL_VAL = 1e-11;
	return (f(v + SMALL_VAL) - f(v - SMALL_VAL)) / (2.*SMALL_VAL);
}

/* 点乘运算 */
template<typename mtt1, typename mtt2>
inline typename mtt1::type do_dot(const int& r, const int& c, const mtt1& mt1, const mtt2& mt2)
{
	using ret_t = typename mtt1::type;
	ret_t ret = 0.0;
	for (int i = 0; i < mt1.c; ++i)
	{
		ret = ret + mt1.template get(r, i) * mt2.template get(i, c);
	}
	return ret;
}

#if 0
template<int row_num1, int col_num1, int row_num2, int col_num2, typename val_t = double>
mat<row_num1, col_num2, val_t> dot(const mat<row_num1, col_num1, val_t>& mt1, const mat<row_num2, col_num2, val_t>& mt2)
{

	using omatt = mat<row_num1, col_num2, val_t>;
	using imatt1 = mat<row_num1, col_num1, val_t>;
	using imatt2 = mat<row_num2, col_num2, val_t>;
	omatt mt_ret;
	for (int r = 0; r < omatt::r; ++r)
	{
		for (int c = 0; c < omatt::c; ++c)
		{
			mt_ret.get(r, c) = do_dot(r, c, mt1, mt2);
		}
	}
	return mt_ret;
}
#endif

/* 加法运算 */
template<typename mtt>
inline typename mtt::type do_add(const int& r, const int& c, const mtt& mt1, const mtt& mt2)
{
	using ret_t = typename mtt::type;
	ret_t ret = 0.0;
	return mt1.get(r, c) + mt2.get(r, c);
}

template<typename mtt>
inline typename mtt::type do_add(const int& r, const int& c, const mtt& mt1, const typename mtt::type& v)
{
	using ret_t = typename mtt::type;
	ret_t ret = 0.0;
	return mt1.get(r, c) + v;
}

template<int row_num, int col_num, typename val_t = double>
mat<row_num, col_num, val_t> operator+(const mat<row_num, col_num, val_t>& mt1, const mat<row_num, col_num, val_t>& mt2)
{
	using omatt = mat<row_num, col_num, val_t>;
	omatt mt_ret;
	for (int r = 0; r < omatt::r; ++r)
	{
		for (int c = 0; c < omatt::c; ++c)
		{
			mt_ret.get(r, c) = do_add(r, c, mt1, mt2);
		}
	}
	return mt_ret;
}

template<int row_num, int col_num, typename val_t = double>
mat<row_num, col_num, val_t> operator+(const val_t& v, const mat<row_num, col_num, val_t>& mt)
{
	using omatt = mat<row_num, col_num, val_t>;
	omatt mt_ret;
	for (int r = 0; r < omatt::r; ++r)
	{
		for (int c = 0; c < omatt::c; ++c)
		{
			mt_ret.get(r, c) = do_add(r, c, mt, v);
		}
	}
	return mt_ret;
}

template<int row_num, int col_num, typename val_t = double>
mat<row_num, col_num, val_t> operator+(const mat<row_num, col_num, val_t>& mt, const val_t& v)
{
	using omatt = mat<row_num, col_num, val_t>;
	omatt mt_ret;
	for (int r = 0; r < omatt::r; ++r)
	{
		for (int c = 0; c < omatt::c; ++c)
		{
			mt_ret.get(r, c) = do_add(r, c, mt, v);
		}
	}
	return mt_ret;
}

/* 减法运算 */
template<typename mtt>
inline typename mtt::type do_minus(const int& r, const int& c, const mtt& mt1, const mtt& mt2)
{
	using ret_t = typename mtt::type;
	ret_t ret = 0.0;
	return mt1.get(r, c) - mt2.get(r, c);
}

template<typename mtt>
inline typename mtt::type do_minus(const int& r, const int& c, const mtt& mt1, const typename mtt::type& v)
{
	using ret_t = typename mtt::type;
	ret_t ret = 0.0;
	return mt1.get(r, c) - v;
}

template<int row_num, int col_num, typename val_t = double>
mat<row_num, col_num, val_t> operator-(const mat<row_num, col_num, val_t>& mt1, const mat<row_num, col_num, val_t>& mt2)
{
	using omatt = mat<row_num, col_num, val_t>;
	omatt mt_ret;
	for (int r = 0; r < omatt::r; ++r)
	{
		for (int c = 0; c < omatt::c; ++c)
		{
			mt_ret.get(r, c) = do_minus(r, c, mt1, mt2);
		}
	}
	return mt_ret;
}

template<int row_num, int col_num, typename val_t = double>
mat<row_num, col_num, val_t> operator-(const val_t& v, const mat<row_num, col_num, val_t>& mt)
{
	using omatt = mat<row_num, col_num, val_t>;
	omatt mt_ret;
	for (int r = 0; r < omatt::r; ++r)
	{
		for (int c = 0; c < omatt::c; ++c)
		{
			mt_ret.get(r, c) = v - mt.get(r, c);
		}
	}
	return mt_ret;
}

template<int row_num, int col_num, typename val_t = double>
mat<row_num, col_num, val_t> operator-(const mat<row_num, col_num, val_t>& mt, const val_t& v)
{
	using omatt = mat<row_num, col_num, val_t>;
	omatt mt_ret;
	for (int r = 0; r < omatt::r; ++r)
	{
		for (int c = 0; c < omatt::c; ++c)
		{
			mt_ret.get(r, c) = do_minus(r, c, mt, v);
		}
	}
	return mt_ret;
}

/* 乘法运算 */
template<int row_num, int col_num, typename val_t = double>
mat<row_num, col_num, val_t> operator*(const val_t& v, const mat<row_num, col_num, val_t>& mt)
{
	using omatt = mat<row_num, col_num, val_t>;
	omatt mt_ret;
	for (int r = 0; r < omatt::r; ++r)
	{
		for (int c = 0; c < omatt::c; ++c)
		{
			mt_ret.get(r, c) = v * mt.get(r, c);
		}
	}
	return mt_ret;
}

template<int row_num, int col_num, typename val_t = double>
mat<row_num, col_num, val_t> operator*(const mat<row_num, col_num, val_t>& mt, const val_t& v)
{
	using omatt = mat<row_num, col_num, val_t>;
	omatt mt_ret;
	for (int r = 0; r < omatt::r; ++r)
	{
		for (int c = 0; c < omatt::c; ++c)
		{
			mt_ret.get(r, c) = mt.get(r, c) * v;
		}
	}
	return mt_ret;
}

template<int row_num, int col_num, typename val_t = double>
mat<row_num, col_num, val_t> operator*(const mat<row_num, col_num, val_t>& mt1, const mat<row_num, col_num, val_t>& mt2)
{
	using omatt = mat<row_num, col_num, val_t>;
	omatt mt_ret;
	for (int r = 0; r < omatt::r; ++r)
	{
		for (int c = 0; c < omatt::c; ++c)
		{
			mt_ret.get(r, c) = mt1.get(r, c) * mt2.get(r, c);
		}
	}
	return mt_ret;
}

/* 除法 */
template<int row_num, int col_num, typename val_t>
mat<row_num, col_num, val_t> operator/(const mat<row_num, col_num, val_t>& mt, const val_t& v)
{
	using omatt = mat<row_num, col_num, val_t>;
	omatt mt_ret;
	for (int r = 0; r < omatt::r; ++r)
	{
		for (int c = 0; c < omatt::c; ++c)
		{
			mt_ret.get(r, c) = mt.get(r, c) / v;
		}
	}
	return mt_ret;
}

template<int row_num, int col_num, typename val_t>
mat<row_num, col_num, val_t> operator/(const val_t& v, const mat<row_num, col_num, val_t>& mt)
{
	using omatt = mat<row_num, col_num, val_t>;
	omatt mt_ret;
	for (int r = 0; r < omatt::r; ++r)
	{
		for (int c = 0; c < omatt::c; ++c)
		{
			mt_ret.get(r, c) = v / mt.get(r, c);
		}
	}
	return mt_ret;
}

template<int row_num, int col_num, typename vt>
mat<row_num, col_num, vt> operator/(const mat<row_num, col_num, vt>& mt1, const mat<row_num, col_num, vt>& mt2)
{
	using omatt = mat<row_num, col_num, vt>;
	omatt mt_ret;
	for (int r = 0; r < omatt::r; ++r)
	{
		for (int c = 0; c < omatt::c; ++c)
		{
			mt_ret.get(r, c) = mt1.get(r, c) / mt2.get(r, c);
		}
	}
	return mt_ret;
}

template<int i1, int i2, typename val_t>
mat<i1, i2, val_t> sqrtl(const mat<i1, i2, val_t>& mt)
{
	using type = mat<i1, i2, val_t>;
	type ret;
	for (int i = 0; i < i1; ++i) 
	{
		for (int j = 0; j < i2; ++j) 
		{
			ret.get(i, j) = sqrtl(mt.get(i,j));
		}
	}
	return ret;
}

template<int row_num, int col_num, typename val_t = double>
mat<row_num, col_num, val_t> sqrtm(const mat<row_num, col_num, val_t>& mt)
{
	using omatt = mat<row_num, col_num, val_t>;
	omatt mt_ret;
	for (int i = 0; i < row_num; ++i)
	{
		for (int j = 0; j < col_num; ++j)
		{
			mt_ret.get(i, j) = sqrtl(mt.get(i, j));
		}
	}
	return mt_ret;
}

/* exp运算 */
template<int row_num, int col_num, typename val_t = double>
mat<row_num, col_num, val_t> exp(const mat<row_num, col_num, val_t>& mt)
{
	using omatt = mat<row_num, col_num, val_t>;
	omatt mt_ret;
	for (int i = 0; i < row_num; ++i)
	{
		for (int j = 0; j < col_num; ++j)
		{
			mt_ret.get(i, j) = exp(mt.get(i, j));
		}
	}
	return mt_ret;
}

/* 卷积运算 */
template<int row_base, int col_base, int row_delta, int col_delta, typename imat_origin, typename imat_tpl>
inline auto col_loop_mul(const imat_origin& mt_origin, const imat_tpl& mt_tpl)
{
	if constexpr (col_delta != 0)
	{
		return mt_origin.template get_val<row_base + row_delta, col_base + col_delta>() * mt_tpl.template get_val<row_delta, col_delta>()
			+ col_loop_mul<row_base, col_base, row_delta, col_delta - 1, imat_origin, imat_tpl>(mt_origin, mt_tpl);
	}
	if constexpr (col_delta == 0)
	{
		return mt_origin.template get_val<row_base + row_delta, col_base + col_delta>() * mt_tpl.template get_val<row_delta, col_delta>();
	}
}

template<int row_base, int col_base, int row_delta, int col_delta, typename imat_origin, typename imat_tpl>
inline auto row_loop_add(const imat_origin& mt_origin, const imat_tpl& mt_tpl)
{
	if constexpr (row_delta != 0)
	{
		return col_loop_mul<row_base, col_base, row_delta, col_delta>(mt_origin, mt_tpl)
			+ col_loop_mul<row_base, col_base, row_delta - 1, col_delta>(mt_origin, mt_tpl);
	}
	if constexpr (row_delta == 0)
	{
		return col_loop_mul<row_base, col_base, row_delta, col_delta>(mt_origin, mt_tpl);
	}
}

template<int r, int c>
struct v_inner_conv
{
	template<typename imat_origin_t, typename imat_tpl_t>
	inline static auto cal(const imat_origin_t& mt_origin, const imat_tpl_t& mt_tpl)
	{
		return row_loop_add<r, c, imat_tpl_t::r - 1, imat_tpl_t::c - 1>(mt_origin, mt_tpl);
	}
};

constexpr int get_step_inner_size(int i_origin, int i_tpl, int i_step)
{
	return (i_origin - i_tpl) / i_step + 1;
}

constexpr int get_pad_size(int i_origin, int i_tpl, int i_step) 
{
	return (((i_origin - i_tpl) / i_step) + (((i_origin - i_tpl) % i_step) == 0 ? 0 : 1))*i_step - (i_origin - i_tpl);
}

constexpr int get_ceil_div(int i_origin, int i_tpl)
{
	return (i_origin / i_tpl + ((i_origin%i_tpl)==0?0:1));
}

template<int input_row, int intput_col, int tpl_row, int tpl_col, int row_step, int col_step>
struct pad_size_t
{
	static constexpr int top = get_pad_size(input_row, tpl_row, row_step) / 2;
	static constexpr int left = get_pad_size(intput_col, tpl_col, col_step) / 2;
	static constexpr int right = get_pad_size(intput_col, tpl_col, col_step) - left;
	static constexpr int bottom = get_pad_size(input_row, tpl_row, row_step) - top;
};

template<int row_step, int col_step, int row_num, int col_num, int tpl_row, int tpl_col, typename val_t>
inline mat<get_step_inner_size(row_num, tpl_row, row_step), get_step_inner_size(col_num, tpl_col, col_step), val_t>
inner_conv(const mat<row_num, col_num, val_t>& mt_origin, const mat<tpl_row, tpl_col, val_t>& mt_tpl)
{
	using ret_type = mat<get_step_inner_size(row_num, tpl_row, row_step), get_step_inner_size(col_num, tpl_col, col_step), val_t>;
	ret_type mt_ret;
	col_loop<ret_type::c - 1, v_inner_conv>(mt_ret, mt_origin, mt_tpl);
	return mt_ret;
}

template<typename mat_t, typename ...mat_ts>
struct st_one_col 
{
	static constexpr int all_size = (mat_t::r*mat_t::c) + st_one_col<mat_ts...>::all_size;
};

template<typename mat_t>
struct st_one_col<mat_t>
{
	static constexpr int all_size = (mat_t::r*mat_t::c);
};

template<typename mat_t, typename ...mat_ts>
void concat_mat(typename mat_t::type* p, const mat_t& mt, const mat_ts... mts) 
{
	constexpr int cpy_size = mat_t::r*mat_t::c;
	//memcpy(p, mt.pval->p, cpy_size*sizeof(mat_t::type));
	for (int i = 0; i < cpy_size; ++i) 
	{
		p[i] = mt.pval->p[i];
	}
	if constexpr(0!=sizeof...(mat_ts))
		concat_mat(p + cpy_size, mts...);
}

template<typename mat_t, typename ...mat_ts>
mat<st_one_col<mat_t, mat_ts...>::all_size, 1> stretch_one_col(const mat_t& mt, const mat_ts&...mts)
{
	using ret_type = mat<st_one_col<mat_t, mat_ts...>::all_size, 1>;
	ret_type ret;
	concat_mat(ret.pval->p, mt, mts...);
	return ret;
}


template<typename mat_t, typename ...mat_ts>
void split_mat(typename mat_t::type* p, const mat_t& mt, const mat_ts... mts)
{
	constexpr int cpy_size = mat_t::r*mat_t::c;
	//memcpy(mt.pval->p, p, cpy_size * sizeof(mat_t::type));
	for (int i = 0; i < cpy_size; ++i)
	{
		mt.pval->p[i] = p[i];
	}
	if constexpr (0 != sizeof...(mat_ts))
		split_mat(p + cpy_size, mts...);
}

template<typename mat_t, typename ...mat_ts>
void split_one_mat(const mat_t& mt, const mat_ts&...mts)
{
	split_mat(mt.pval->p, mts...);
}

template<int i1, int i2, typename val_t>
mat<i1, i2, val_t> abs(const mat<i1, i2, val_t>& mt)
{
	mat<i1, i2, val_t> ret;
	for (int i = 0; i < i1; ++i) 
	{
		for (int j = 0; j < i2; ++j) 
		{
			ret.get(i, j) = abs(mt.get(i, j));
		}
	}
	return ret;
}

#include "ht_memory.h"

template<typename val_t>
void write_file(const val_t& vt, ht_memory& mry) 
{
	mry << vt;
}

template<int row_num, int col_num, typename val_t>
void write_file(const mat<row_num, col_num, val_t>& mt, ht_memory& mry)
{
	for (int r = 0; r < row_num; ++r)
	{
		for (int c = 0; c < col_num; ++c)
		{
			write_file(mt.get(r, c), mry);
		}
	}
}

template<typename val_t>
void read_file(ht_memory& mry, val_t& vt) 
{
	mry >> vt;
}

template<int row_num, int col_num, typename val_t>
void read_file(ht_memory& mry, mat<row_num, col_num, val_t>& mt)
{
	for (int r = 0; r < row_num; ++r)
	{
		for (int c = 0; c < col_num; ++c)
		{
			read_file(mry, mt.get(r, c));
		}
	}
}


#endif
