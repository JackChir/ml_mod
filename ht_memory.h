#pragma once
#include <memory.h>
#include <stdlib.h>

class ht_memory 
{
public:
	enum endian 
	{
		big_endian = 0,
		little_endian
	};
	enum strategy 
	{
		buf_stable,								// 静态的存储区，不可扩展
		buf_flexable,								// 可扩展的存储区，当写空间不够时自动扩展
	};
protected:
	unsigned char*			m_sz_buf;				// 指向的内存区
	unsigned int				m_u_buf_len;				// 内存区长度
	mutable unsigned int		m_u_read_idx;				// 下一个将读到的位置
	unsigned int				m_u_write_idx;			// 下次将写到的位置
	
	endian					m_e_endian;				// m_sz_buf内存端序
	strategy					m_e_strategy;				// 内存管理策略

	unsigned int				m_u_expand_size;			// 单次的扩充长度
protected:
	virtual void* ht_realloc(void* p, const unsigned int& u_expected_size);
	virtual void ht_free(void* p);
public:
	ht_memory(const endian& e_endian, const unsigned int& u_expand_size = 512);
	virtual ~ht_memory();

	/* 复制操作 */
	ht_memory(const ht_memory& other);
	ht_memory& operator=(const ht_memory& other);
	void clone(const ht_memory& other);				// 从输入克隆出一个新的可扩充对象
	void get_buf_from(ht_memory& other);			// 获取other的内存，并使other失去其对内存的控制

	/* 读移动操作 */
	void skip(const unsigned int& u_len) const;
	void operator+=(const unsigned int& u_len) const;
	ht_memory& operator++();
	void reset_read();

	/* 查询操作 */
	unsigned char* origin_buf() const;
	unsigned char* buf() const;
	unsigned int size() const;
	unsigned char& operator[](const unsigned int& idx) const;
	unsigned int read_size() const;
	unsigned int write_size() const;

	/* 读写操作 */
	unsigned int read(char* sz_buf, const unsigned int& u_len) const;
	unsigned int write(const char* sz_buf, const unsigned int& u_len);
	void reset();
	void load(void* p, const unsigned int& u_len, const strategy& e_strategy);
	void cload(const void* p, const unsigned int& u_len);

	/* 修改对象 */
	void trim_read();
	void* abort_memory(unsigned int& u_read_idx, unsigned int& u_write_idx);
	void set_capacity(const unsigned int& u_buf_len);

	/* 适用于容器 */
	template<typename T>
	unsigned int read(T& t, const unsigned int& u_len) const 
	{
		unsigned int u_left = m_u_write_idx - m_u_read_idx;
		unsigned int u_read_len = u_left < u_len ? u_left : u_len;
		typedef typename T::value_type ElemType;
		ElemType ele;
		for (unsigned int i = 0; i < u_read_len; ++i)
		{
			operator>>(ele);
			t.push_back(ele);
		}
		return u_read_len;
	}
	int read_file(const char* cstr_file_path);
	int write_file(const char* cstr_file_path);

	template<typename T>
	ht_memory& operator<<(const T& t) 
	{
		T t1(t);
		swap_endian(t1, m_e_endian);
		/* 判断长度是否超标 */
		unsigned int u_expected_size = m_u_write_idx + sizeof(t);
		u_expected_size = (u_expected_size / m_u_expand_size + ((u_expected_size%m_u_expand_size != 0) ? 1 : 0))*m_u_expand_size;
		if (u_expected_size > m_u_buf_len)
		{
			if (m_e_strategy == buf_stable)
			{
				return *this;
			}
			else if (m_e_strategy == buf_flexable)
			{
				unsigned char* p = reinterpret_cast<unsigned char*>(ht_realloc(m_sz_buf, u_expected_size));
				if (u_expected_size == 0) 
				{
					/* 原指针失效，属于未定义行为，可能是程序有问题 */
					return *this;
				}
				if (!p)
				{
					/* 内存分配失败，原空间依然有效，但是内存已经不足 */
					return *this;
				}
				m_sz_buf = p;
				m_u_buf_len = u_expected_size;
			}
			else {}
		}
		/* 在write位置写入数据 */
		unsigned char* p_w = m_sz_buf + m_u_write_idx;
		memcpy(p_w, &t1, sizeof(t1));
		m_u_write_idx += sizeof(t1);
		return *this;
	}

	template<typename T>
	const ht_memory& operator>>(T& t) const 
	{
		if (m_u_write_idx < m_u_read_idx + sizeof(t)) 
		{
			return *this;
		}
		memcpy(&t, m_sz_buf + m_u_read_idx, sizeof(t));
		swap_endian(t, m_e_endian);
		m_u_read_idx += sizeof(t);
		return *this;
	}

	template<typename T>
	bool try_get(T& t) const
	{
		if (m_u_write_idx < m_u_read_idx + sizeof(t)) 
		{
			return false;
		}
		operator>>(t);
		return true;
	}

	template<typename T>
	bool try_read(T& t, const unsigned int& u_len) const
	{
		if (m_u_write_idx < m_u_read_idx + u_len)
		{
			return false;
		}
		read(t, u_len);
		return true;
	}
};


ht_memory::endian system_endian();

inline void swap_array(unsigned char* sz_buf, const unsigned int& u_buf_len)
{
	unsigned int u_loop_count = u_buf_len / 2;
	for (unsigned int u_cnt = 0; u_cnt < u_loop_count; ++u_cnt)
	{
		unsigned char uc_tmp = sz_buf[u_cnt];
		sz_buf[u_cnt] = sz_buf[u_buf_len - 1u - u_cnt];
		sz_buf[u_buf_len - 1u - u_cnt] = uc_tmp;
	}
}

template<typename T>
void swap_endian(T& t, const ht_memory::endian& e)
{
}

#define DECLARE_SWAP_TYPE(T) inline void swap_endian(T& t, const ht_memory::endian& e) \
{ \
	if (system_endian() == e) \
	{ \
		return; \
	} \
	swap_array(reinterpret_cast<unsigned char*>(&t), sizeof(t)); \
}

DECLARE_SWAP_TYPE(short)
DECLARE_SWAP_TYPE(unsigned short)
DECLARE_SWAP_TYPE(int)
DECLARE_SWAP_TYPE(unsigned int)
DECLARE_SWAP_TYPE(long)
DECLARE_SWAP_TYPE(unsigned long)

#if defined(_WIN32) || defined(_WIN64)
#	if(_MSC_VER >= 1600)
DECLARE_SWAP_TYPE(long long)
#	endif
#endif
