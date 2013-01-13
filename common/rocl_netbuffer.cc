#include "rocl_common.h"

rocl::net_buffer::net_buffer(std::size_t hint)
	: m_v(), _rpos(0), _wpos(0)
	  
{
	if (hint)
		m_v.reserve(hint);
}

std::size_t rocl::net_buffer::size() const
{
	return m_v.size();
}

std::size_t rocl::net_buffer::rpos() const
{
	return _rpos;
}

std::size_t rocl::net_buffer::wpos() const
{
	return size();
}

std::uint8_t rocl::net_buffer::read8()
{
	if (_rpos-0 >= size()) 
		throw read_ex();
	return m_v[_rpos++];
}


