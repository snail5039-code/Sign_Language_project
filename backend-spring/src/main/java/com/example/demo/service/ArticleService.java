package com.example.demo.service;

import java.util.List;

import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.server.ResponseStatusException;

import com.example.demo.dao.ArticleDao;
import com.example.demo.dto.Article;

@Service
public class ArticleService {
	
	private ArticleDao articleDao;
	public ArticleService(ArticleDao articleDao) {
		this.articleDao = articleDao;
	}
	
	public void write(Article article) {
		this.articleDao.write(article);
	}

	public List<Article> articleList() {
		return this.articleDao.articleList();
	}


	public Article articleDetail(int id) {
		return this.articleDao.articleDetail(id);
	}

	public void articleModify(Article article) {
		 this.articleDao.articleModify(article);	
	}

	public void articleDelete(int id) {
		int rows = this.articleDao.articleDelete(id);
		if(rows == 0) {
			// http상태코드를  의도적으로 실패로 만들기 위한 예외
			throw new ResponseStatusException(HttpStatus.NOT_FOUND, "해당 게시글은 존재하지 않습니다.");
		} 

	}
}
