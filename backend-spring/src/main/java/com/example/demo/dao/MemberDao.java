package com.example.demo.dao;

import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

import com.example.demo.dto.Member;

@Mapper
public interface MemberDao {

	@Insert("""
			insert into member(regDate, updateDate, loginId, loginPw, name, email, countryId)
			values (now(), now(), #{loginId}, #{loginPw}, #{name}, #{email}, #{countryId})
			""")
	void join(Member member);

	@Select("""
			select *
				from member
				where loginId = #{loginId}
			""")
	Member findByLoginId(String loginId);

	@Select("""
			select *
				from member
				where loginId = #{loginId}
				 and loginPw = #{loginPw}
			""")
	Member findByLoginIdAndPw(Member member);
	
	

}
