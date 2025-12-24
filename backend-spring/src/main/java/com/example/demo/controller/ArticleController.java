package com.example.demo.controller;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.ibatis.annotations.Delete;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.example.demo.dto.Article;
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
	
	@PostMapping("/boards")
	public Map<String, Object> write(@RequestBody Article article) {
		this.articleService.write(article);
		return Map.of("message", "작성완료");
	}
	
	@GetMapping("/list")
	public Map<String, Object> list() {
	List<Article> articles = this.articleService.articleList();
	
	Map<String, Object> result = new HashMap<>();
	result.put("articles", articles);
	
	return result;
	}
	
	@GetMapping("/detail")
	public Map<String, Object> detail(int id) {
		Article article = this.articleService.articleDetail(id);
		Map<String, Object> result = new HashMap<>();
		result.put("article", article);
		
		return result;
	}
	
	@PutMapping("/modify")
	public Map<String, Object> modify(@PathVariable int id,@RequestBody Article article) {
		this.articleService.articleModify(article);
		Map<String, Object> result = new HashMap<>();
		result.put("message", "수정완료");
		
		return result;
	}
	
	@DeleteMapping("/delete")
	public Map<String, Object> delete(@PathVariable int id) {
		this.articleService.articleDelete(id);
		Map<String, Object> result = new HashMap<>();
		result.put("message", "삭제완료");
		
		return result;
	}
}
