package com.example.demo.dao;

import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface ArticleDao {
	
	@Insert("""
			insert into article
				set regDate = now()
					,updateDate = now()
					,title = #{title}
					,content = #{content}
					,boardId = #{boardId}
			""")
	void write(int boardId, String title, String content);

}
