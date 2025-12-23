package com.example.demo.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.example.demo.service.ArticleService;

import io.swagger.v3.oas.annotations.parameters.RequestBody;

@RestController
@CrossOrigin(origins = "http://localhost:5173")
@RequestMapping("/api")
public class ArticleController {
	
	private ArticleService articleService;
	public ArticleController(ArticleService articleService) {
		this.articleService = articleService;
	}
	
	@GetMapping("/boards")
	public String write(int boardId, String title, String content) {
		this.articleService.write(boardId, title, content);
		return "글 작성 완료";
	}
	
	

}
