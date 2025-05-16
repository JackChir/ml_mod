#ifndef _BASE_LOGIC_HPP_
#define _BASE_LOGIC_HPP_

/* ����������� */

template<int r, int c, template<int, int> class op, typename omatt, typename... imatts>
inline void row_loop(omatt& omt, const imatts&...imts)
{
	omt.template get_val<r, c>() = op<r, c>::cal(imts...);
	if constexpr (r != 0)
	{
		row_loop<r - 1, c, op>(omt, imts...);
	}
}

template<int c, template<int, int> class op, typename omatt, typename...imatts>
inline void col_loop(omatt& omt, const imatts&...imts)
{
	row_loop<omt.r - 1, c, op>(omt, imts...);
	if constexpr (c != 0)
	{
		col_loop<c - 1, op>(omt, imts...);
	}
}

#endif
