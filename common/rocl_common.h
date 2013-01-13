#if !defined (__ROCL_COMMON_H__)
#define __ROCL_COMMON_H__ 1

#include <cstdint>
#include <vector>

namespace rocl {

	class read_ex {
	};

	class net_buffer {
		typedef std::vector<std::uint8_t> vec_type;
		vec_type m_v;
		vec_type::size_type _rpos;
	public:
		net_buffer(std::size_t blocksize);
		std::size_t size() const;
		std::size_t read_pos() const;
		std::size_t write_pos() const;
		const std::uint8_t* data() const;
		std::uint8_t read8();
		std::uint16_t read16();
		std::uint32_t read32();
		std::uint64_t read64();
		net_buffer& write(std::uint8_t v);
		net_buffer& write(std::uint16_t v);
		net_buffer& write(std::uint32_t v);
		net_buffer& write(std::uint64_t v);
	};

	namespace protocol {
		
		typedef std::uint64_t seq_id;
		typedef std::uint32_t req_len;

		enum req_id {
			
		};

		enum data_flags {
			LITTLE_ENDIAN = 0,
			BIG_ENDIAN = 1
		};


		class flags {
			std::uint32_t _f;
		public:
			flags(data_flags f);
			std::uint32_t val() const;
		};

		class header {
			std::uint32_t _req_id;
			std::uint32_t _req_len;
			std::uint32_t _req_id;
			std::uint32_t _flags;
		public:
			header(std::uint32_t rid, 
			       std::uint32_t len,
			       req_id id,
			       flags f);
			// conversion from network byte order.
			header(const std::uint32_t* buf,
			       std::size_t len);
		};

		class request : public header {
		public:
			request(std::uint32_t rid,
				std::uint32_t len,
				req_id id,
				flags f);
			virtual ~request();
		};

		class response : public header {
		public:
			response(std::uint32_t rid,
				 std::uint32_t len,
				 req_id id,
				 flags f);
			virtual ~response();
		};

	}

	namespace common {
		
		

	}
}


// Local variables:
// mode: c++
// end:
#endif
