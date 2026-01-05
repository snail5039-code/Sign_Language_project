package com.example.demo.help;

import com.example.demo.help.HelpCardDtos.ChatRequest;
import com.example.demo.help.HelpCardDtos.ChatResponse;
import com.example.demo.help.HelpCardDtos.HelpCard;
import com.example.demo.openai.OpenAiService;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

import java.util.List;

@RestController
@RequestMapping("/api/help")
@CrossOrigin(origins = {"http://localhost:5173", "http://localhost:5174"})
public class HelpController {

    private final HelpCardService service;
    private final OpenAiService openAi;

    public HelpController(HelpCardService service, OpenAiService openAi) {
        this.service = service;
        this.openAi = openAi;
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

        String category = req != null && req.context != null ? nz(req.context.category) : "";
        String message  = req != null ? nz(req.message) : "";
        String lastQ    = req != null && req.context != null ? nz(req.context.lastQuestionType) : "";

        String msgRaw = message == null ? "" : message.trim();

        // 1) 빈 입력
        if (msgRaw.isBlank()) {
            return mk("cards", "응, 무슨 일이야?", List.of(), "ASK_PROBLEM_TYPE");
        }

        // 2) 안전 필터(최소)
        String low = msgRaw.toLowerCase();
        if (containsAny(low, "자살", "죽고", "죽을", "목숨", "끝내고")) {
            return mk(
                    "cards",
                    "지금은 네 안전이 제일 중요해. 혼자 버티지 말고 주변 도움을 꼭 받아.",
                    List.of(),
                    "SAFETY_CHECK"
            );
        }

        // 3) AI에게 전부 판단 맡김 (히스토리 포함)
        OpenAiService.DialogPlan plan;
        try {
            List<ChatRequest.HistoryItem> hist = (req != null ? req.history : null);
            plan = openAi.dialogPlan(msgRaw, category, lastQ, hist, 5);
        } catch (Exception e) {
            plan = null;
        }

        if (plan == null || plan.text == null || plan.text.isBlank()) {
            return mk(
                    "cards",
                    "잠깐 오류가 있었어. 방금 말한 걸 한 번만 더 보내줘!",
                    List.of(),
                    "ASK_PROBLEM_TYPE"
            );
        }

        String intent = (plan.intent == null || plan.intent.isBlank()) ? "PROBLEM" : plan.intent;
        String nextQ  = (plan.nextQuestionType == null || plan.nextQuestionType.isBlank())
                ? "ASK_FOLLOWUP"
                : plan.nextQuestionType;

        // 4) 잡담/짜증이면 카드 없이 자연스럽게
        //    (stateEnded=true도 여기서 카드 끊는 용도로 같이 사용)
        if (plan.stateEnded || "CHITCHAT".equals(intent) || "FRUSTRATION".equals(intent)) {
            return mk(
                    "cards",
                    plan.text,
                    List.of(),
                    nextQ
            );
        }

        // 5) 문제/환경힌트면 카드 추천
        var rr = service.recommend(category, msgRaw, 3);
        var ids = rr.cards.stream().map(c -> c.id).toList();

        return mk(
                "cards",
                plan.text,
                ids,
                nextQ
        );
    }

    private static String nz(String s) { return s == null ? "" : s; }

    private static boolean containsAny(String hay, String... needles) {
        if (hay == null) return false;
        for (String n : needles) if (hay.contains(n)) return true;
        return false;
    }

    private static ChatResponse mk(String type, String text, List<String> matched, String nextQ) {
        ChatResponse r = new ChatResponse();
        r.type = type;
        r.text = text;
        r.matched = matched;
        r.nextQuestionType = nextQ;
        return r;
    }
}
