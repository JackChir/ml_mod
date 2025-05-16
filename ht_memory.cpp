#include <exception>
#include <stdexcept>
#include <string>
#include <fstream>
#include "ht_memory.h"

#define nullptr 0

union endian_check
{
	unsigned short us;
	unsigned char	 sz[2];
};

union endian_check const g_ec = {0x0001};

inline unsigned int get_trimmed_size(const unsigned int& u_expect_size, const unsigned int& u_trim_size) 
{
	return (u_expect_size / u_trim_size + u_expect_size % u_trim_size == 0 ? 0 : 1)*u_trim_size;
}

ht_memory::endian system_endian() 
{
	return g_ec.sz[0] == 0x01 ? ht_memory::little_endian : ht_memory::big_endian;
}

ht_memory::ht_memory(const endian& e_endian, const unsigned int& u_expand_size)
	:m_sz_buf(nullptr)
	, m_u_buf_len(0u)
	, m_u_read_idx(0u)
	, m_u_write_idx(0u)
	, m_e_endian(e_endian)
	, m_e_strategy(buf_flexable)
	, m_u_expand_size((u_expand_size==0||u_expand_size>1024*1024*128)?512:u_expand_size)
{
}

ht_memory::ht_memory(const ht_memory & other)
	:m_sz_buf(nullptr)
	, m_u_buf_len(other.m_u_buf_len)
	, m_u_read_idx(other.m_u_read_idx)
	, m_u_write_idx(other.m_u_write_idx)
	, m_e_endian(other.m_e_endian)
	, m_e_strategy(other.m_e_strategy)
	, m_u_expand_size(other.m_u_expand_size)
{
	/* 如果是固定内存策略的则赋值指针即可，如果是可扩展存储区则复制存储区 */
	if (other.m_e_strategy == buf_stable) 
	{
		m_sz_buf = other.m_sz_buf;
	}
	else if (other.m_e_strategy == buf_flexable) 
	{
		m_sz_buf = reinterpret_cast<unsigned char*>(ht_realloc(m_sz_buf, other.m_u_buf_len));
		memset(m_sz_buf, 0, other.m_u_buf_len);
		memcpy(m_sz_buf, other.m_sz_buf, other.m_u_buf_len);
	}
}

ht_memory & ht_memory::operator=(const ht_memory & other)
{
	if (m_sz_buf)
	{
		ht_free(m_sz_buf);
	}
	m_sz_buf = nullptr;
	/* 进一步进行处理 */
	m_u_buf_len = other.m_u_buf_len;
	m_u_read_idx = other.m_u_read_idx;
	m_u_write_idx = other.m_u_write_idx;
	m_e_endian = other.m_e_endian;
	m_e_strategy = other.m_e_strategy;
	m_u_expand_size = other.m_u_expand_size;
	if (other.m_e_strategy == buf_stable)
	{
		m_sz_buf = other.m_sz_buf;
	}
	else if (other.m_e_strategy == buf_flexable)
	{
		m_sz_buf = reinterpret_cast<unsigned char*>(ht_realloc(m_sz_buf, other.m_u_buf_len));
		memset(m_sz_buf, 0, other.m_u_buf_len);
		memcpy(m_sz_buf, other.m_sz_buf, other.m_u_buf_len);
	}
	return *this;
}

void ht_memory::clone(const ht_memory & other)
{
	ht_free(m_sz_buf);
	m_sz_buf = nullptr;
	m_e_strategy = buf_flexable;
	m_sz_buf = reinterpret_cast<unsigned char*>(ht_realloc(m_sz_buf, other.m_u_buf_len));
	memset(m_sz_buf, 0, other.m_u_buf_len);
	memcpy(m_sz_buf, other.m_sz_buf, other.m_u_buf_len);
	m_u_buf_len = other.m_u_buf_len;
	m_u_read_idx = other.m_u_read_idx;
	m_u_write_idx = other.m_u_write_idx;
	m_e_endian = other.m_e_endian;
	//m_e_strategy = other.m_e_strategy;
	m_u_expand_size = other.m_u_expand_size;
}

void ht_memory::get_buf_from(ht_memory & other)
{
	ht_free(m_sz_buf);
	m_u_buf_len = m_u_read_idx = m_u_write_idx = 0u;
	m_u_buf_len = other.m_u_buf_len;
	m_e_endian = other.m_e_endian;
	m_e_strategy = other.m_e_strategy;
	m_u_expand_size = other.m_u_expand_size;
	m_sz_buf = reinterpret_cast<unsigned char*>(other.abort_memory(m_u_read_idx, m_u_write_idx));
}

ht_memory::~ht_memory()
{
	if (m_sz_buf) 
	{
		ht_free(m_sz_buf);
	}
}

void * ht_memory::ht_realloc(void* p, const unsigned int& u_expected_size)
{
	if (m_e_strategy == buf_flexable)
	{
		void* p1 = realloc(p, u_expected_size);
		return p1;
	}
	return nullptr;
}

void ht_memory::ht_free(void * p)
{
	if (p && m_e_strategy == buf_flexable)
		free(p);
}

unsigned char & ht_memory::operator[](const unsigned int & idx) const
{
	if (idx >= m_u_buf_len) 
	{
		throw std::runtime_error(std::string("ht_memory::operator[](const unsigned int&) const越界访问"));
	}
	return m_sz_buf[idx];
}

unsigned int ht_memory::read_size() const
{
	return m_u_read_idx;
}

unsigned int ht_memory::write_size() const
{
	return m_u_write_idx;
}

unsigned int ht_memory::size() const
{
	if (m_u_read_idx > m_u_write_idx) 
	{
		throw std::runtime_error("ht_memory::size已读长度超过已写长度");
	}
	return m_u_write_idx - m_u_read_idx;
}

void ht_memory::load(void * p, const unsigned int & u_len, const strategy& e_strategy)
{
	m_e_strategy = e_strategy;
	m_u_write_idx = u_len;
	m_u_read_idx = 0;
	if (e_strategy == buf_flexable)
	{
		/* 判断长度是否超标 */
		if (u_len > m_u_buf_len)
		{
			unsigned char* p = reinterpret_cast<unsigned char*>(ht_realloc(m_sz_buf, u_len));
			if (u_len == 0)
			{
				/* 原指针失效，属于未定义行为，可能是程序有问题 */
				return;
			}
			if (!p)
			{
				/* 内存分配失败，原空间依然有效，但是内存已经不足 */
				return;
			}
			memset(p, 0, u_len);
			m_sz_buf = p;
			m_u_buf_len = u_len;
		}
		/* 在write位置写入数据 */
		memcpy(m_sz_buf, p, u_len);
		m_u_write_idx = u_len;
	}
	else 
	{
		m_sz_buf = (unsigned char*)p;
		m_u_write_idx = u_len;
		m_u_buf_len = u_len;
	}
}

void ht_memory::cload(const void * p, const unsigned int & u_len)
{
	m_e_strategy = buf_flexable;
	m_u_write_idx = u_len;
	m_u_read_idx = 0;
	/* 判断长度是否超标 */
	if (u_len > m_u_buf_len)
	{
		unsigned char* p = reinterpret_cast<unsigned char*>(ht_realloc(m_sz_buf, u_len));
		if (u_len == 0)
		{
			/* 原指针失效，属于未定义行为，可能是程序有问题 */
			return;
		}
		if (!p)
		{
			/* 内存分配失败，原空间依然有效，但是内存已经不足 */
			return;
		}
		memset(p, 0, u_len);
		m_sz_buf = p;
		m_u_buf_len = u_len;
	}
	/* 在write位置写入数据 */
	memcpy(m_sz_buf, p, u_len);
	m_u_write_idx = u_len;
}

void ht_memory::skip(const unsigned int & u_len) const
{
	if (m_u_write_idx < m_u_read_idx + u_len)
	{
		m_u_read_idx = m_u_write_idx;
		return;
	}
	m_u_read_idx += u_len;
}

void ht_memory::operator+=(const unsigned int & u_len) const
{
	skip(u_len);
}

ht_memory & ht_memory::operator++()
{
	skip(1);
	return *this;
}

void ht_memory::reset_read()
{
	m_u_read_idx = 0u;
}

unsigned char * ht_memory::buf() const
{
	return m_sz_buf + m_u_read_idx;
}

unsigned int ht_memory::read(char * sz_buf, const unsigned int & u_len) const
{
	unsigned int u_left = m_u_write_idx - m_u_read_idx;
	unsigned int u_read_len = u_left < u_len ? u_left : u_len;
	memcpy(sz_buf, m_sz_buf + m_u_read_idx, u_read_len);
	m_u_read_idx += u_read_len;
	return u_read_len;
}

unsigned int ht_memory::write(const char * sz_buf, const unsigned int & u_len)
{
	/* 判断长度是否超标 */
	unsigned int u_expected_size = m_u_write_idx + u_len;
	u_expected_size = (u_expected_size / m_u_expand_size + ((u_expected_size%m_u_expand_size != 0) ? 1 : 0))*m_u_expand_size;
	if (u_expected_size > m_u_buf_len)
	{
		if (m_e_strategy == buf_stable)
		{
			return 0;
		}
		else if (m_e_strategy == buf_flexable)
		{
			unsigned char* p = reinterpret_cast<unsigned char*>(ht_realloc(m_sz_buf, u_expected_size));
			if (u_expected_size == 0)
			{
				/* 原指针失效，属于未定义行为，可能是程序有问题 */
				return 0;
			}
			if (!p)
			{
				/* 内存分配失败，原空间依然有效，但是内存已经不足 */
				return 0;
			}
			m_sz_buf = p;
			m_u_buf_len = u_expected_size;
		}
		else {}
	}
	/* 在write位置写入数据 */
	unsigned char* p_w = m_sz_buf + m_u_write_idx;
	memcpy(p_w, sz_buf, u_len);
	m_u_write_idx += u_len;
	return 0;
}

void ht_memory::reset()
{
	if (m_sz_buf)
		memset(m_sz_buf, 0, m_u_buf_len);
	m_u_read_idx = 0;
	m_u_write_idx = 0;
}

void ht_memory::trim_read()
{
	memmove(m_sz_buf, buf(), size());
	m_u_write_idx -= m_u_read_idx;
	m_u_read_idx = 0u;
	memset(m_sz_buf+m_u_write_idx, 0, m_u_buf_len - m_u_write_idx);
	unsigned int u_free_len = m_u_buf_len - m_u_write_idx;
	if (u_free_len > m_u_expand_size && m_e_strategy == buf_flexable) 
	{
		m_u_buf_len = get_trimmed_size(m_u_write_idx, m_u_expand_size);
		m_sz_buf = reinterpret_cast<unsigned char*>(ht_realloc(m_sz_buf, m_u_buf_len));
	}
}

void * ht_memory::abort_memory(unsigned int & u_read_idx, unsigned int & u_write_idx)
{
	trim_read();
	void* p_ret = m_sz_buf;
	u_read_idx = m_u_read_idx, u_write_idx = m_u_write_idx;
	m_sz_buf = nullptr;
	m_u_buf_len = m_u_read_idx = m_u_write_idx = 0u;
	return p_ret;
}

void ht_memory::set_capacity(const unsigned int & u_buf_len)
{
	unsigned int u_expect_len = m_u_read_idx + u_buf_len;
	if (m_u_buf_len > u_expect_len || m_e_strategy == buf_stable) 
	{
		return;
	}
	unsigned int u_len = get_trimmed_size(u_expect_len, m_u_expand_size);
	m_sz_buf = reinterpret_cast<unsigned char*>(ht_realloc(m_sz_buf, u_len));
	m_u_buf_len = u_len;
}

int ht_memory::read_file(const char* cstr_file_path)
{
	std::ifstream ifs(cstr_file_path, std::ifstream::binary);
	if (!ifs.is_open()) 
	{
		return -1;
	}
	ifs.seekg(0, ifs.end);
	auto length = ifs.tellg();
	ifs.seekg(0, ifs.beg);
	unsigned int u_expected_size = length;
	u_expected_size = (u_expected_size / m_u_expand_size + ((u_expected_size%m_u_expand_size != 0) ? 1 : 0))*m_u_expand_size;
	m_sz_buf = (unsigned char*)ht_realloc(m_sz_buf, u_expected_size);
	m_u_buf_len = u_expected_size;
	memset(m_sz_buf, 0, m_u_buf_len);
	m_u_read_idx = 0;
	m_u_write_idx = length;
	ifs.read((char*)m_sz_buf, length);
	ifs.close();
	return 0;
}

int ht_memory::write_file(const char * cstr_file_path)
{
	std::ofstream ofs(cstr_file_path, std::ofstream::trunc | std::ofstream::binary);
	if (!ofs.is_open()) 
	{
		return -1;
	}
	ofs.write(const_cast<const char*>(reinterpret_cast<char*>(buf())), size());
	ofs.flush();
	ofs.close();
	return 0;
}
