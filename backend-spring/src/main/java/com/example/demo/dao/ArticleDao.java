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
			insert into article
				set regDate = now()
					,updateDate = now()
					,title = #{title}
					,content = #{content}
					,boardId = #{boardId}
			""")
	void write(Article article);
	
	
	@Select("""
			select * from article
			order by id desc
			""")
	List<Article> articleList();

	@Select("""
			select * from article
			where id = #{id}
			""")
	Article articleDetail(int id);

	@Update("""
			update article
				set updateDate = now()
					,title = #{title}
					,content = #{content}
				where id = #{id}
			""")
	void articleModify(Article article);

	@Delete("""
			delete from article
			where id = #{id}
			""")
	int articleDelete(int id);
	

}
