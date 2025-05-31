#ifndef _MAT_HPP_
#define _MAT_HPP_
#include <memory>
#include <climits>
#include <float.h>


#include <map>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>

#include "ht_memory.h"


template<typename val_t>
val_t max_and_swap(const val_t& v1, const val_t& v2)
{
	return v1 < v2 ? v2 : v1;
}

template<typename type>
struct destoryer 
{
	static void do_destory(type& v)
	{
		//printf("%s\r\n", typeid(type).name());
	}
};

template<int i_size, typename val_t>
struct mat_m
{
	val_t sz_ele[i_size];
	val_t* p;

	mat_m() :p(nullptr)
	{
		p = sz_ele;
		for (int i = 0; i < i_size; ++i)
		{
			new(p + i) val_t(0);
		}
	}
	~mat_m()
	{
		if (p)
		{
			p = 0;
		}
	}
	val_t& get(const int& len_1d, const int& i_1d_idx, const int& i_2d_idx)
	{
		val_t& ret = p[i_2d_idx + len_1d * i_1d_idx];
		return ret;
	}

	val_t max_abs() const
	{
		double d = -1*DBL_MAX;
		for (int i = 0; i < i_size; ++i) 
		{
			d = d < abs(p[i]) ? abs(p[i]) : d;
		}
		return d;
	}

	val_t max() const
	{
		val_t d = -1 * DBL_MAX;
		for (int i = 0; i < i_size; ++i)
		{
			//d = d < (p[i]) ? (p[i]) : d;
			d = max_and_swap(p[i], d);
		}
		return d;
	}

	val_t sum() const 
	{
		val_t d_sum = 0.;
		for (int i = 0; i < i_size; ++i)
		{
			d_sum = d_sum + p[i];
		}
		return d_sum;
	}

	template<int len_1d, int i_1d_idx, int i_2d_idx>
	inline val_t& get_val()
	{
		static_assert((i_2d_idx + len_1d * i_1d_idx) < i_size, "ERROR:mat_m over flow?????????");
		return p[i_2d_idx + len_1d * i_1d_idx];
	}

	template<int len_1d, int i_1d_idx, int i_2d_idx>
	inline val_t get_val() const
	{
		return p[i_2d_idx + len_1d * i_1d_idx];
	}
};


template<int row_num, int col_num, typename val_t = double>
struct mat
{
	using t_type = mat<col_num, row_num, val_t>;
	using type = val_t;
	typedef val_t vt;
	static constexpr int r = row_num;
	static constexpr int c = col_num;
	using mat_m_t = mat_m<row_num * col_num, val_t>;
	std::shared_ptr<mat_m_t> pval;
	bool b_t;

	mat():b_t(false)
	{
		pval = std::make_shared<mat_m_t>();
	}
	mat(const mat<row_num, col_num, val_t>& other) :pval(other.pval), b_t(other.b_t)
	{
	}
	~mat() 
	{
		pval.reset();
	}

	mat(const val_t&& v):b_t(false)
	{
		pval = std::make_shared<mat_m_t>();
		for (int i = 0; i < row_num; ++i)
		{
			for (int j = 0; j < col_num; ++j)
			{
				pval->get(col_num, i, j) = v;
			}
		}
	}
	mat(const val_t& v) :b_t(false)
	{
		pval = std::make_shared<mat_m_t>();
		for (int i = 0; i < row_num; ++i)
		{
			for (int j = 0; j < col_num; ++j)
			{
				pval->get(col_num, i, j) = v;
			}
		}
	}
#if 0
	template<typename val_other_t>
	mat(const val_other_t& v) :b_t(false)
	{
		pval = std::make_shared<mat_m_t>();
		for (int i = 0; i < row_num; ++i)
		{
			for (int j = 0; j < col_num; ++j)
			{
				pval->get(col_num, i, j) = static_cast<val_t>(v);
			}
		}
	}
#endif
	mat(const std::initializer_list<val_t>& lst):b_t(false)
	{
		pval = std::make_shared<mat_m_t>();
		auto itr = lst.begin();
		for (int i = 0; i < row_num; ++i)
		{
			for (int j = 0; j < col_num; ++j)
			{
				if (itr == lst.end())return;
				pval->get(col_num, i, j) = *itr;
				itr++;
			}
		}
	}

	val_t& get(const int& i_row, const int& i_col)
	{
		if (!b_t)
			return pval->get(col_num, i_row, i_col);
		else
			return pval->get(row_num, i_col, i_row);
	}

	val_t get(const int& i_row, const int& i_col) const
	{
		if (!b_t)
			return pval->get(col_num, i_row, i_col);
		else
			return pval->get(row_num, i_col, i_row);
	}

	template<int i_1d_idx, int i_2d_idx>
	inline val_t& get_val()
	{
		if (!b_t)
			return pval->template get_val<col_num, i_1d_idx, i_2d_idx>();
		else
			return pval->template get_val<row_num, i_2d_idx, i_1d_idx>();
	}

	template<int i_1d_idx, int i_2d_idx>
	inline val_t get_val() const
	{
		static_assert(i_1d_idx < row_num && i_2d_idx < col_num, "ERROR: mat::get_val overflow!!!!!");
		if (!b_t)
			return pval->template get_val<col_num, i_1d_idx, i_2d_idx>();
		else
			return pval->template get_val<row_num, i_2d_idx, i_1d_idx>();
	}

	mat<col_num, row_num, val_t> t() const
	{
		mat<col_num, row_num, val_t> ret;
		ret.pval = pval;
		ret.b_t = !b_t;
		return ret;
	}

	val_t max_abs() const
	{
		return pval->max_abs();
	}

	val_t max() const 
	{
		return pval->max();
	}

	val_t sum() const 
	{
		return pval->sum();
	}

	void print() const
	{
		std::cout << "[" << std::endl;
		for (int i = 0; i < row_num; ++i)
		{
			std::cout << std::setw(3) << "[";
			for (int j = 0; j < col_num; ++j)
			{
				std::cout << (j != 0 ? "," : "") << std::setw(10) << get(i, j);
			}
			std::cout << std::setw(3) << "]" << std::endl;
		}
		std::cout << "]" << std::endl;
	}

	template<int other_col_num>
	mat<row_num, other_col_num, val_t> dot(const mat<col_num, other_col_num, val_t>& mt) const
	{
		using omatt = mat<row_num, other_col_num, val_t>;
		omatt mt_ret;
		for (int r = 0; r < omatt::r; ++r)
		{
			for (int c = 0; c < omatt::c; ++c)
			{
				mt_ret.get(r, c) = do_dot(r, c, *this, mt);
			}
		}
		return mt_ret;
	}

	mat<row_num, col_num, val_t> rot180() const
	{
		mat<row_num, col_num, val_t> ret;
		for (int r = 0; r < row_num; ++r) 
		{
			for (int c = 0; c < col_num; ++c)
			{
				ret.get(r, c) = get(row_num-1-r, col_num-1-c);
			}
		}
		return ret;
	}

	template<int row_base, int col_base, int row_num_other, int col_num_other>
	void assign(const mat<row_num_other, col_num_other, val_t>& mt_other) 
	{
		/* ???????????????????????????? */
		for (int r = 0; r < row_num_other; ++r) 
		{
			for (int c = 0; c < col_num_other; ++c) 
			{
				if (r + row_base < 0 || c + col_base < 0)
				{
					continue;
				}
				if (r + row_base >= row_num || c + col_base >= col_num) 
				{
					break;
				}
				get(r + row_base, c + col_base) = mt_other.get(r, c);
			}
		}
	}

	template<int top_pad, int left_pad, int right_pad, int bottom_pad>
	mat<row_num + top_pad + bottom_pad, col_num + left_pad + right_pad, val_t>
		pad() const
	{
		using mat_ret_t = mat<row_num + top_pad + bottom_pad, col_num + left_pad + right_pad, val_t>;
		mat_ret_t mt_ret;
		mt_ret.template assign<top_pad, left_pad>(*this);
		return mt_ret;
	}

	template<int row_span, int col_span>
	mat<row_num + row_span*(row_num - 1), col_num + col_span*(col_num-1)>
		span() const
	{
		using mat_ret_t = mat<row_num + row_span * (row_num - 1), col_num + col_span * (col_num - 1)>;
		mat_ret_t mt_ret;
		for (int r = 0; r < row_num; ++r) 
		{
			for (int c = 0; c < col_num; ++c) 
			{
				mt_ret.get(r*(row_span + 1), c*(col_span + 1)) = get(r, c);
			}
		}
		return mt_ret;
	}

	template<int row_base, int col_base, int row_len, int col_len>
	val_t region_max(int& i_row, int& i_col) const 
	{
		static_assert(row_base < row_num && col_base < col_num, "region_max overflow!!!");
		val_t d_max = -1. * DBL_MAX;
		for (int r = row_base; r < row_base + row_len && r < row_num; ++r) 
		{
			for (int c = col_base; c < col_base + col_len && c < col_num; ++c) 
			{
				if (d_max < get(r, c)) 
				{
					i_row = r, i_col = c;
					d_max = get(r, c);
				}
			}
		}
		return d_max;
	}

	mat<row_num*col_num, 1, val_t> one_col() const 
	{
		mat<row_num*col_num, 1, val_t> ret;
		ret.pval = pval;
		return ret;
	}

	template<typename t>
	static void print_sub_type(t) 
	{
		std::cout << typeid(t).name();
	}

	template<int r, int c, typename t>
	static void print_sub_type(mat<r, c, t>)
	{
		mat<r, c, t>::print_type();
	}

	static void print_type() 
	{
		printf("<matrix %d * %d type: ", row_num, col_num);
		print_sub_type<val_t>(val_t());
		printf(">\r\n");
	}

	const val_t& operator[](const int& idx) const
	{
		return pval->p[idx];
	}

	val_t& operator[](const int& idx)
	{
		return pval->p[idx];
	}

	mat<row_num, col_num, val_t>& operator+=(const mat<row_num, col_num, val_t>& other)
	{
		for (int i = 0; i < row_num; ++i)
		{
			for (int j = 0; j < col_num; ++j)
			{
				get(i, j) += other.get(i, j);
			}
		}
		return *this;
	}

	friend std::ostream& operator<<(std::ostream& ofs, const mat<row_num, col_num, val_t>& mt)
	{
		std::cout << "[" ;
		for (int i = 0; i < row_num; ++i)
		{
			std::cout << std::setw(3) << "[";
			for (int j = 0; j < col_num; ++j)
			{
				std::cout << (j != 0 ? "," : "") << mt.get(i, j);
			}
			std::cout << std::setw(3) << "]";
		}
		std::cout << "]";
		return ofs;
	}

	int size() const
	{
		return row_num * col_num;
	}

	mat<row_num - 1, col_num - 1, val_t> algebraic_complement_val(const int& r, const int& c) const
	{
		mat<row_num - 1, col_num - 1, val_t> mt_ret;
		int retr = 0, retc = 0;
		for (int mr = 0; mr < row_num; mr++)
		{
			if (mr == r) continue;
			for (int mc = 0; mc < col_num; mc++)
			{
				if (mc == c) continue;
				mt_ret.get(retr, retc++) = get(mr, mc);
			}
			retc = 0;
			retr++;
		}
		return mt_ret;
	}
};

template<typename type>
struct mat_size
{
	static constexpr int num = 1;
};

template<int target_col, int target_row, typename target_val_t>
struct mat_size<mat<target_col, target_row, target_val_t> >
{
	static constexpr int num = target_col * target_row * mat_size<target_val_t>::num;
};

template<typename unite_type>
struct mat_unite_type 
{
	using type = unite_type;
};

template<int row_num, int col_num, typename val_t>
struct mat_unite_type<mat<row_num, col_num, val_t> >
{
	using type = typename mat_unite_type<val_t>::type;
};

template<typename mat_t>
struct unite_mat
{
	using type = mat<mat_size<mat_t>::num, 1, typename mat_unite_type<mat_t>::type>;
};

template<typename mat_t>
using unite_mat_t = typename unite_mat<mat_t>::type;

template<typename target_type>
struct stretch_to_unite
{
	template<typename type1>
	static target_type cal(type1& v)
	{
		auto k = v.one_col();
		return stretch_to_unite<target_type>::cal(k);
	}

	static target_type cal(target_type& v)
	{
		return v;
	}
};

// ??mt_max?mt_optional???mt_max?mt_optional?????
template<int i1, int i2, typename val_t>
mat<i1, i2, val_t> max_and_swap(const mat<i1, i2, val_t>& mt_max, const mat<i1, i2, val_t>& mt_optional)
{
	using ret_type = mat<i1, i2, val_t>;
	ret_type ret;
	for (int i = 0; i < i1; ++i) 
	{
		for (int j = 0; j < i2; ++j) 
		{
			ret.get(i, j) = max_and_swap(mt_max.get(i,j), mt_optional.get(i, j));
		}
	}
	return ret;
}

template<typename val_t>
val_t max_and_choose(const val_t& v1, const val_t& v2, const val_t& v3, const val_t& v4)
{
	return v1 < v2 ? v3 : v4;
}

// ??mt_judget1?mt_judge2??????mt_judge1??mt_judge2????mt_optional1?????mt_optional2
template<int i1, int i2, typename val_t>
mat<i1, i2, val_t> max_and_choose(const mat<i1, i2, val_t>& mt_judge1, const mat<i1, i2, val_t>& mt_judge2, const mat<i1, i2, val_t>& mt_optional1, const mat<i1, i2, val_t>& mt_optional2)
{
	using ret_type = mat<i1, i2, val_t>;
	ret_type ret;
	for (int i = 0; i < i1; ++i)
	{
		for (int j = 0; j < i2; ++j)
		{
			ret.get(i, j) = max_and_choose(mt_judge1.get(i, j), mt_judge2.get(i, j), mt_optional1.get(i, j), mt_optional2.get(i, j));
		}
	}
	return ret;
}

template<int row_num, int col_num, typename val_t>
struct destoryer<mat<row_num, col_num, val_t> >
{
	using type = mat<row_num, col_num, val_t>;
	static void do_destory(type& v)
	{
		v.~type();
	}
};


// vec_cur??????????mt???????
// ??mt?3*3???vec_cur?{0, 1, 2}????mt[0][0]*mt[1][1]*mt[2][2]
template<int N>
double sub_mul_cal(const mat<N, N, double>& mt, const std::vector<int>& vec_cur)
{
	double dmul = 1.;
	for (size_t siz_add_itr = 0; siz_add_itr < vec_cur.size(); ++siz_add_itr)
	{
		dmul *= mt.get(siz_add_itr, vec_cur[siz_add_itr]);
		//printf("%4.2lf ", mt.get(siz_add_itr, vec_cur[siz_add_itr]));
	}
	//printf("=%4.2lf", dmul);
	return dmul;
}


// 使用逆序数法求行列式
template<int N>
void det_cal(double& dret, const mat<N, N, double>& mt, const std::vector<int>& vec_idx, const size_t& siz_cur, const double& dflag = 1.)
{
	//printf("\n%4.2lf ", dflag);
	dret += (sub_mul_cal(mt, vec_idx)*dflag);			// * ??????
	for (size_t siz_itr = siz_cur; siz_itr < vec_idx.size() - 1 ; ++siz_itr)
	{
		std::vector<int> vec_cur(vec_idx);
		double dcurflag = dflag;
		for (size_t siz_in_itr = siz_itr + 1; siz_in_itr < vec_cur.size(); ++siz_in_itr)
		{
			std::swap(vec_cur[siz_itr], vec_cur[siz_in_itr]);
			dcurflag *= -1.;
			//dret += (sub_mul_cal(mt, vec_cur)*dflag);
			det_cal(dret, mt, vec_cur, siz_itr + 1, dcurflag);
		}
	}
}

// 求行列式的值
template<int N>
double det(const mat<N, N, double>& mt)
{
	std::vector<int> vec(N, 0);
	int idx = 0;
	std::generate(vec.begin(), vec.end(), [&idx]()
	{
		return idx++;
	});
	double dret = 0.;
	det_cal(dret, mt, vec, 0, 1.);
	return dret;
}

template<int N, typename val_t>
mat<N, N, val_t> algebraic_complement(const mat<N, N, val_t>& mt)
{
	mat<N, N, val_t> mtret;
	double drflag = 1.;
	for (int i = 0; i < N; ++i)
	{
		double dcflag = 1.;
		for (int j = 0; j < N; ++j)
		{
			mtret.get(i, j) = (drflag * dcflag * det(mt.algebraic_complement_val(i, j)));
			dcflag *= -1.;
		}
		drflag *= -1.;
	}
	return mtret.t();
}

template<int N, typename val_t>
mat<N, N, val_t> inverse(const mat<N, N, val_t>& mt)
{
	return algebraic_complement(mt)/det(mt);
}

template<typename cur_mt_t, typename ...other_mt_t>
constexpr int row_sum()
{
	if constexpr (sizeof...(other_mt_t) == 0)
	{
		return cur_mt_t::r;
	}
	else
	{
		return cur_mt_t::r + row_sum<other_mt_t...>();
	}
}

template<int begin_row, typename ret_mt_t, typename cur_mt_t, typename ...other_mt_t>
void __join_col(ret_mt_t& mt_ret, const cur_mt_t& mt_cur, const other_mt_t& ...mt_other)
{
	for (int i = 0; i < cur_mt_t::r; ++i)
	{
		for (int j = 0; j < cur_mt_t::c; ++j)
		{
			mt_ret.get(i + begin_row, j) = mt_cur.get(i, j);
		}
	}
	if constexpr (sizeof...(other_mt_t) > 0)
	{
		__join_col<begin_row + cur_mt_t::r, ret_mt_t, other_mt_t...>(mt_ret, mt_other...);
	}
}


template<typename cur_mt_t, typename ...mat_ts>
mat<row_sum<cur_mt_t,mat_ts...>(), cur_mt_t::c> join_col(const cur_mt_t& mt, const mat_ts& ...mts)
{
	using ret_type = mat<row_sum<cur_mt_t,mat_ts...>(), cur_mt_t::c>;
	ret_type mt_ret;
	__join_col<0, ret_type, cur_mt_t, mat_ts...>(mt_ret, mt, mts...);
	return mt_ret;
}

template<typename cur_mt_t, typename ...mat_ts>
constexpr int col_num()
{
	if constexpr (sizeof...(mat_ts) == 0)
	{
		return cur_mt_t::c;
	}
	else
	{
		return cur_mt_t::c + col_num<mat_ts...>();
	}
}

template<int begin_col, typename ret_mt_t, typename cur_mt_t, typename ...mat_ts>
void __join_row(ret_mt_t& mt_ret, const cur_mt_t& mt_cur, const mat_ts& ...mts)
{
	for (int i = 0; i < cur_mt_t::r; ++i)
	{
		for (int j = 0; j < cur_mt_t::c; ++j)
		{
			mt_ret.get(i, j + begin_col) = mt_cur.get(i, j);
		}
	}
	if constexpr (sizeof...(mts) > 0)
	{
		__join_row<begin_col + cur_mt_t::c, ret_mt_t, mat_ts...>(mt_ret, mts...);
	}
}

template<typename cur_mt_t, typename ...mat_ts>
mat<cur_mt_t::r, col_num<cur_mt_t, mat_ts...>()> join_row(const cur_mt_t& mt, const mat_ts& ...mts)
{
	using ret_type = mat<cur_mt_t::r, col_num<cur_mt_t, mat_ts...>()>;
	ret_type mt_ret;
	__join_row<0, ret_type, cur_mt_t, mat_ts...>(mt_ret, mt, mts...);
	return mt_ret;
}

#include <vector>
/* 输出改成0均值1均方差的 */
template<int row_num, int col_num, typename val_t>
inline std::vector<mat<row_num, col_num, val_t> > normalize(const std::vector<mat<row_num, col_num, val_t> >& vec_input, mat<row_num, col_num, val_t>& mt_mean, mat<row_num, col_num, val_t>& mt_div)
{
	if (1 == vec_input.size())return vec_input;
	using type = mat<row_num, col_num, val_t>;
	std::vector<type> vec_ret;
	//type mt_mean, mt_div;
	for (int i = 0; i < vec_input.size(); ++i)
	{
		mt_mean = mt_mean + vec_input[i] / static_cast<val_t>(vec_input.size());
	}

	for (int i = 0; i < vec_input.size(); ++i)
	{
		auto delta = vec_input[i] - mt_mean;
		mt_div = mt_div + delta * delta / static_cast<val_t>(vec_input.size());
	}

	auto mt_s = sqrtl(mt_div);

	for (int i = 0; i < vec_input.size(); ++i)
	{
		vec_ret.push_back((vec_input[i] - mt_mean) / mt_s);
	}
	return vec_ret;
}

// 对行进行归一化
template<int row_num, int col_num, typename val_t>
inline mat<row_num, col_num, val_t> normalize(const mat<row_num, col_num, val_t>& mt_input, mat<row_num, col_num, val_t>& mt_mean, mat<row_num, col_num, val_t>& mt_sqrt)
{
	mat<1, row_num, val_t> mt_one(1.0);
	mt_mean = mt_one.t().dot((mt_one.dot(mt_input) / static_cast<val_t>(row_num)));
	auto delta = mt_input - mt_mean;
	mt_sqrt = mt_one.t().dot(sqrtl(mt_one.dot((delta*delta)) / static_cast<val_t>(row_num)))+ val_t(1e-10); // 加上一个小的数值避免除0
	return (mt_input - mt_mean) / mt_sqrt; // 加上一个小的数值避免除0
}

// 把第一个矩阵向左移动并将第二个矩阵插入到第一个矩阵的右边
template<int row_num, int col_num, int insert_col, typename val_t>
void move_left_and_insert(mat<row_num, col_num, val_t>& mt_input, const mat<row_num, insert_col, val_t>& mt_insert)
{
	static_assert(insert_col <= col_num, "ERROR: move_left_and_insert insert_col overflow");
	for (int i = 0; i < row_num; ++i)
	{
		for (int j = 0; j < col_num - insert_col; ++j)
		{
			mt_input.get(i, j) = mt_input.get(i, j + insert_col);
		}
		for (int j = col_num - insert_col; j < col_num; ++j)
		{
			mt_input.get(i, j) = mt_insert.get(i, j - (col_num - insert_col));
		}
	}
}

template<int row_num, int col_num, typename val_t>
inline void inplace_RoPE(const val_t& x11, const val_t& x12, const val_t& x21, const val_t& x22, mat<row_num, col_num, val_t>& mt_input)
{
	for (int i = 0; i < row_num; ++i)
	{
		for (int j = 0; j < col_num; ++j)
		{
			val_t x1 = mt_input.get(i, j);
			val_t x2 = mt_input.get(j, i);
			mt_input.get(i, j) = x11 * x1 + x12 * x2;
			mt_input.get(j, i) = x21 * x1 + x22 * x2;
		}
	}
}

template<int row_num, int col_num, typename val_t>
inline mat<row_num, col_num, val_t> RoPE(const val_t& x11, const val_t& x12, const val_t& x21, const val_t& x22, const mat<row_num, col_num, val_t>& mt_input)
{
	mat<row_num, col_num, val_t> mt_ret(mt_input);
	inplace_RoPE(x11, x12, x21, x22, mt_ret);
	return mt_ret;
}

#endif
