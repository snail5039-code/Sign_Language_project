package com.example.demo.controller;

import java.util.List;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;
import com.example.demo.dto.DictionaryDto.DictionaryItem;
import com.example.demo.service.DictionaryService;
import lombok.RequiredArgsConstructor;

@RestController
@RequiredArgsConstructor
@CrossOrigin(origins = "http://localhost:5173")
@RequestMapping("/api/dictionary")
public class DictionaryController {
	
	private final DictionaryService dictionaryService;
	
	@GetMapping
	public List<DictionaryItem> list(@RequestParam(defaultValue = "") String q) {
		return dictionaryService.search(q);
	}
	
	@GetMapping("/{id}")
	public DictionaryItem detail(@PathVariable String id) {
		DictionaryItem item = dictionaryService.detail(id);
		if (item == null) throw new ResponseStatusException(HttpStatus.NOT_FOUND, "not found");
		return item;
	}
}