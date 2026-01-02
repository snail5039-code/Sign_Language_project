package com.example.demo.help;

import com.example.demo.help.HelpCardDtos.ChatRequest;
import com.example.demo.help.HelpCardDtos.ChatResponse;
import com.example.demo.help.HelpCardDtos.HelpCard;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

import java.util.List;

@RestController
@RequestMapping("/api/help")
@CrossOrigin(origins = "http://localhost:5173")
public class HelpController {

    private final HelpCardService service;

    public HelpController(HelpCardService service) {
        this.service = service;
    }

    @GetMapping("/categories")
    public List<String> categories() {
        return service.categories();
    }

    @GetMapping("/cards")
    public List<HelpCard> list(
            @RequestParam(defaultValue = "") String category,
            @RequestParam(defaultValue = "") String q
    ) {
        return service.list(category, q);
    }

    @GetMapping("/cards/{id}")
    public HelpCard detail(@PathVariable String id) {
        HelpCard c = service.get(id);
        if (c == null) throw new ResponseStatusException(HttpStatus.NOT_FOUND, "not found");
        return c;
    }

    @PostMapping("/chat")
    public ChatResponse chat(@RequestBody ChatRequest req) {
        String category = (req != null && req.context != null) ? req.context.category : "";
        String message = (req != null) ? req.message : "";

        var rec = service.recommend(category, message, 3);

        ChatResponse res = new ChatResponse();
        res.type = "cards";
        res.text = "관련 해결 방법을 찾았어! 아래 카드부터 확인해봐.";
        res.matched = rec.stream().map(c -> c.id).toList();
        return res;
    }
}
