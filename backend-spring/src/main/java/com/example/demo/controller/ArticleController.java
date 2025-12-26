package com.example.demo.controller;

import java.util.List;
import java.util.Map;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

import com.example.demo.dto.Article;
import com.example.demo.service.ArticleService;

import jakarta.servlet.http.HttpSession;


@CrossOrigin(origins = "http://localhost:5173")
@RestController
@RequestMapping("/api")
public class ArticleController {

    private final ArticleService articleService;
    public ArticleController(ArticleService articleService) {
        this.articleService = articleService;
    }

    @PostMapping("/boards")
    public Map<String, Object> write(@RequestBody Article article, HttpSession session) {
    	
    	Integer memberId = (Integer) session.getAttribute("loginedMemberId");
    	
    	if (article.getBoardId() ==  null) {
    		throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "boardId is required");
    	}
    	if (memberId ==  null) {
    		throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "memberId is required");
    	}
    	
    	article.setMemberId(memberId);
        articleService.write(article);
        return Map.of("message", "작성완료");
    }

    @GetMapping("/boards")
    public List<Article> list() {
        return articleService.articleList();
    }

    @GetMapping("/boards/{id}")
    public Article detail(@PathVariable int id) {
        return articleService.articleDetail(id);
    }

    @PutMapping("/boards/{id}")
    public Map<String, Object> modify(@PathVariable int id,
        @RequestBody Article article) {
        article.setId(id);
        articleService.articleModify(article);
        return Map.of("message", "수정완료");
    }

    @DeleteMapping("/boards/{id}")
    public Map<String, Object> delete(@PathVariable int id) {
        articleService.articleDelete(id);
        return Map.of("message", "삭제완료");
    }
}
