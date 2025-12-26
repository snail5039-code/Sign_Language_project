package com.example.demo.dao;

import java.util.List;

import org.apache.ibatis.annotations.Delete;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import com.example.demo.dto.Article;

@Mapper
public interface ArticleDao {
	
	@Insert("""
	        INSERT INTO article (regDate, updateDate, title, content, boardId, memberId)
	        VALUES (NOW(), NOW(), #{title}, #{content}, #{boardId}, #{memberId})
	        """)
	int write(Article article);
	
	
	@Select("""
	        SELECT * FROM article
	        WHERE boardId = #{boardId}
	        ORDER BY id DESC
	        """)
	List<Article> articleListByBoardId(int boardId);

	@Select("""
	        SELECT * FROM article
	        ORDER BY id DESC
	        """)
	List<Article> articleList();

	 @Select("""
		     SELECT * FROM article
		     WHERE id = #{id}
		     """)
	Article articleDetail(int id);

	@Update("""
			UPDATE article
				SET updateDate = NOW()
					,title = #{title}
					,content = #{content}
					, boardId = #{boardId}
				WHERE id = #{id}
			""")
	void articleModify(Article article);

	@Delete("""
			DELETE FROM article
			WHERE id = #{id}
			""")
	int articleDelete(int id);
	

}
