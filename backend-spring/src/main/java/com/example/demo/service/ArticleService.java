package com.example.demo.service;

import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.GetMapping;

import com.example.demo.dao.ArticleDao;

@Service
public class ArticleService {
	
	private ArticleDao articleDao;
	public ArticleService(ArticleDao articleDao) {
		this.articleDao = articleDao;
	}
	
	public void write(int boardId, String title, String content) {
		this.articleDao.write(boardId, title, content);
	}
	

	
	
}
