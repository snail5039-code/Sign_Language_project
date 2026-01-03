package com.example.demo.service;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;
import com.example.demo.config.KcisaProperties;
import com.example.demo.dto.DictionaryDto.DictionaryItem;
import com.example.demo.dto.DictionaryDto.Media;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class DictionaryService {

    private final KcisaProperties props;          // baseUrl, serviceKey, numOfRows
    private final RestTemplate restTemplate = new RestTemplate();
    private final ObjectMapper om = new ObjectMapper();

    /** ✅ 검색(목록) */
    public List<DictionaryItem> search(String q) {
        String keyword = (q == null) ? "" : q.trim();
        int rows = Math.min(Math.max(props.getNumOfRows(), 1), 100);

        return fetchPage(1, rows, keyword);
    }

    /**
     * KCISA가 localId로 “단건 조회 파라미터”가 명확히 없는 경우가 많아서,
     * (1) id를 keyword로 검색해보고,
     * (2) 없으면 1~몇페이지 훑어서 localId 일치 찾는 방식으로 안전하게 처리
     */
    public DictionaryItem detail(String id) {
        if (id == null || id.isBlank()) return null;

        // 1) id를 keyword로 검색해서 바로 매칭되는지
        List<DictionaryItem> byKeyword = fetchPage(1, 50, id);
        for (DictionaryItem it : byKeyword) {
            if (id.equals(it.id())) return fixVideoIfNeeded(it);
        }

        // 2) fallback: 1~5 페이지 훑기 (데모/개발용)
        for (int page = 1; page <= 5; page++) {
            List<DictionaryItem> pageItems = fetchPage(page, 100, "");
            for (DictionaryItem it : pageItems) {
                if (id.equals(it.id())) return fixVideoIfNeeded(it);
            }
        }

        return null;
    }

    // 내부 KCISA 호출 + 매핑

    private List<DictionaryItem> fetchPage(int pageNo, int numOfRows, String keyword) {
        String url = UriComponentsBuilder
            .fromUriString(props.getBaseUrl()) // 예: https://api.kcisa.kr/API_CNV_054/request
            .queryParam("serviceKey", props.getServiceKey())
            .queryParam("pageNo", pageNo)
            .queryParam("numOfRows", numOfRows)
            // 빈 값이어도 포함시키는 게 안전
            .queryParam("keyword", keyword == null ? "" : keyword)
            .queryParam("collectionDb", "")
            .build()
            .encode(java.nio.charset.StandardCharsets.UTF_8)
            .toUriString();

        HttpHeaders headers = new HttpHeaders();
        headers.setAccept(List.of(MediaType.APPLICATION_JSON)); // JSON 받기
        HttpEntity<Void> entity = new HttpEntity<>(headers);

        ResponseEntity<String> res = restTemplate.exchange(url, HttpMethod.GET, entity, String.class);

        try {
            JsonNode root = om.readTree(res.getBody());
            JsonNode itemNode = root.path("response").path("body").path("items").path("item");

            List<JsonNode> nodes = new ArrayList<>();
            if (itemNode.isArray()) itemNode.forEach(nodes::add);
            else if (itemNode.isObject()) nodes.add(itemNode);
            else return Collections.emptyList();

            List<DictionaryItem> out = new ArrayList<>();
            for (JsonNode n : nodes) {
                // 키 이름은 실제 응답에 따라 다를 수 있어 fallback 여러 개 둠
                String id = pick(n, "localId", "LOCAL_ID", "id");
                String word = pick(n, "title", "TITLE", "word");
                String category = pick(n, "categoryType", "collectionDb", "CATEGORY", "category");
                String description = pick(n, "description", "DESCRIPTION", "meaning");
                String sub = pick(n, "subDescription", "SUB_DESCRIPTION");
                
                boolean subIsUrl = isUrl(sub);
                
                // meaning: description 우선, 없으면 sub(단, sub가 URL이면 meaning으로 쓰지 않음)
                String meaning = !description.isBlank() ? description : (!subIsUrl ? sub : "");
                
                // examples(예문) sub가 URL이면 예문으로 넣지 말기, 문장일 경우만 1줄 예문처럼
                List<String> examples = 
                		(!subIsUrl && !sub.isBlank() && !sub.equals(meaning))
                			? List.of(sub)
                			: List.of();
                
                // videourl: 영상파일링크만 인정
                //String videoUrl = isVideoFileUrl(rawUrl) ? rawUrl : "";
                
                String signImages = pick(n, "signImages", "SIGN_IMAGES");
                String gifUrl = firstCsv(signImages);
                
                String rawUrl = pick(n, "url", "URL", "videoUrl");
                
                String candidate = "";
                if (isUrl(rawUrl)) candidate = rawUrl;
                if (candidate.isBlank() && subIsUrl) candidate = sub;
                
                String videoUrl = "";
                
                if (isUrl(candidate)) {
                	if (isImageFileUrl(candidate)) {
                		if (gifUrl.isBlank()) gifUrl = candidate;
                	} else {
                		videoUrl = candidate;
                	}
                }             
                //if (gifUrl.isBlank()) gifUrl = pick(n, "imageObject", "IMAGE_OBJECT", "gifUrl");
                
                out.add(new DictionaryItem (
                		id,
                		word,
                		category,
                		meaning,
                		examples,
                		new Media(videoUrl, gifUrl)
                		));
            }

            return out;

        } catch (Exception e) {
            throw new RuntimeException("KCISA parse fail: " + e.getMessage(), e);
        }
    }

    private String pick(JsonNode n, String... keys) {
        for (String k : keys) {
            JsonNode v = n.get(k);
            if (v != null && !v.isNull()) {
                String s = v.asText("");
                if (!s.isBlank()) return s;
            }
        }
        return "";
    }

    private String firstCsv(String s) {
        if (s == null) return "";
        String t = s.trim();
        if (t.isBlank()) return "";
        int idx = t.indexOf(',');
        return (idx < 0) ? t : t.substring(0, idx).trim();
    }
    
    private boolean isUrl(String s) {
    	if (s == null) return false;
    	String t = s.trim().toLowerCase();
    	return t.startsWith("http://") || t.startsWith("https://");
    }
    
    private boolean isVideoFileUrl(String s) {
    	if (s == null) return false;
    	String t = s.trim().toLowerCase();
    	if (!(t.startsWith("http://") || t.startsWith("https://"))) return false;
    	
    	return t.contains(".mp4") || t.contains(".webm") || t.contains(".mov") || t.contains(".m4v");
    }
    
    private boolean isImageFileUrl(String s) {
    	if (s == null) return false;
    	String t = s.trim().toLowerCase();
    	return t.contains(".jpg") || t.contains(".jpeg") || t.contains(".png") || t.contains(".gif") || t.contains(".webp");
    }
    
    private String extractVideoFromHtml(String pageUrl, String thumbUrl) {
        if (pageUrl == null || pageUrl.isBlank()) return "";

        try {
            // ✅ UA 넣어서 HTML 가져오기
            HttpHeaders h = new HttpHeaders();
            h.set(HttpHeaders.USER_AGENT, "Mozilla/5.0");
            h.setAccept(List.of(MediaType.TEXT_HTML));

            ResponseEntity<String> r = restTemplate.exchange(
                pageUrl, HttpMethod.GET, new HttpEntity<>(h), String.class
            );

            String html = r.getBody();
            if (html == null || html.isBlank()) return "";

            // ✅ 1) 절대경로 video 링크 찾기
            String norm = html
            		.replace("\\/", "/")
            		.replace("&amp;", "&");
            
            var p1 = java.util.regex.Pattern.compile(
            	"(https?://[^\"'\\s>]+\\.(mp4|mov|m4v|webm|m3u8)(\\?[^\"'\\s>]*)?)",
                java.util.regex.Pattern.CASE_INSENSITIVE
            );

            var m1 = p1.matcher(norm);
            if (m1.find()) return m1.group(1);

            // ✅ 2) 상대경로 /multimedia/.. video 링크 찾기
            var p2 = java.util.regex.Pattern.compile(
            	"(/[^\"'\\s>]+\\.(mp4|mov|m4v|webm|m3u8)(\\?[^\"'\\s>]*)?)",
                java.util.regex.Pattern.CASE_INSENSITIVE
            );

            var m2 = p2.matcher(norm);
            while (m2.find()) {
            	String path = m2.group(1);
            	if (path.contains("/multimedia/")) {
            		return "https://sldict.korean.go.kr" + path;
            	}
            }

        } catch (Exception e) {
        	// 디버깅용
        	e.printStackTrace();
        }

        //  3) HTML에서 못 찾으면 썸네일로 mp4 추측
        return guessMp4FromThumb(thumbUrl);
    }

    private String guessMp4FromThumb(String thumbUrl) {
        if (thumbUrl == null || thumbUrl.isBlank()) return "";

        String u = thumbUrl.trim();

        // MOV 썸네일일 때만 시도(IMG는 이미지일 가능성 큼)
        if (!u.contains("/MOV")) return "";

        u = u.replaceAll("_[0-9]+X[0-9]+\\.(jpg|jpeg|png|gif|webp)(\\?.*)?$", ".mp4");
        return u;
    }

    private DictionaryItem fixVideoIfNeeded(DictionaryItem it) {
        if (it == null || it.media() == null) return it;

        String v = it.media().videoUrl();
        if (v == null || v.isBlank()) return it;
        
        v = v.replace("http://sldict.korean.go.kr", "https://sldict.korean.go.kr");
        if (isVideoFileUrl(v)) return it;

        // 페이지 링크면 -> HTML에서 mp4 추출 시도
        
        String extracted = extractVideoFromHtml(v, it.media().gifUrl());
        if (extracted != null && !extracted.isBlank()) {
        	return new DictionaryItem(
        			it.id(),
        			it.word(),
        			it.category(),
        			it.meaning(),
        			it.examples(),
        			new Media(extracted, it.media().gifUrl())
        			);
        }
        
        return it;
    }

}