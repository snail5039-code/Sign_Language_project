package com.example.demo.controller;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.server.ResponseStatusException;
import com.example.demo.dto.TranslateResponse;
import com.example.demo.dto.TranslationLog;
import com.example.demo.service.TranslateResponseService;

@RestController
@CrossOrigin(origins = "http://localhost:5173")
public class TranslateController {
	//파이썬 연결
	private final WebClient webClient = WebClient.create("http://127.0.0.1:8000");
	private final TranslateResponseService translateResponseService;
	
	public TranslateController(TranslateResponseService translateResponseService) {
		this.translateResponseService = translateResponseService;
	}
	
    @PostMapping("/api/translate")
    public TranslateResponse translate(@RequestBody Map<String, Object> body) {
    	
		TranslateResponse res = webClient.post()
    				.uri("/predict")
    				.bodyValue(body)
    				.retrieve()
    				.bodyToMono(TranslateResponse.class)
    				.timeout(Duration.ofSeconds(3))
    				.onErrorReturn(new TranslateResponse(null, "번역 실패", 0.0, 0))
    				.block();
    		
    		if(res == null) {
    			res = new TranslateResponse(null, "번역 실패", 0.0, 0);
    		}
    		double conf = Math.max(0.0, Math.min(1.0, res.getConfidence()));
            res.setConfidence(conf);
			
    	this.translateResponseService.save(res.getText(), res.getConfidence());
    	
    	return res;
    }
    
    private static final List<DictionaryItem> MOCK = List.of(
    		new DictionaryItem(
    				"hello", "안녕하세요", "인사",
    				"처음 만나거나 인사할 때 쓰는 표현",
    				List.of("안녕하세요. 만나서 반가워요."),
    				new Media("/media/hello.mp4", "/media/hello.gif")
    				),
    		new DictionaryItem(
    				"thanks", "감사합니다", "인사",
    				"고마움을 표현할 때 쓰는 말",
    				List.of("도와줘서 감사합니다.."),
    				new Media("", "")
    				),
    		new DictionaryItem(
    				"help", "도와주세요", "응급",
    				"도움이 필요할 때 요청하는 표현",
    				List.of("도와주세요!"),
    				new Media("", "")
    			)
    		);
    
    @GetMapping("/api/dictionary")
    public List<DictionaryItem> List(@RequestParam(defaultValue = "") String q) {
    	if (q == null || q.isBlank()) return MOCK;
    	
    	String keyword = q.trim();
    	return MOCK.stream()
    			.filter(x -> x.word().contains(keyword) || x.meaning().contains(keyword))
    			.toList();
    }
    
    @GetMapping("/api/dictionary/{id}")
    public DictionaryItem detail(@PathVariable String id) {
    	return MOCK.stream()
    			.filter(x -> x.id().equals(id))
    			.findFirst()
    			.orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "not found"));
    }
    
    public record Media(String videoUrl, String gifUrl) {}
    
    public record DictionaryItem(
    		String id,
    		String word,
    		String category,
    		String meaning,
    		List<String> examples,
    		Media media
    		) {}
    		
    
    //디비 조회하기 귀찮으니 만드는 거
    @GetMapping("api/translation-log")
    public List<TranslationLog> recent(@RequestParam(defaultValue = "10") int limit) {
    	if(limit < 1) limit = 1;
    	if(limit > 100) limit = 100;
    	
    	return translateResponseService.findRecent(limit);
    }
}
