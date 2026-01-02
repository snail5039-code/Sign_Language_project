package com.example.demo.help;

import com.example.demo.help.HelpCardDtos.HelpCard;
import com.example.demo.help.HelpCardDtos.HelpCardsFile;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;
import java.io.InputStream;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class HelpCardService {

    private final ObjectMapper om;

    private List<HelpCard> cards = new ArrayList<>();
    private Map<String, HelpCard> byId = new HashMap<>();

    public HelpCardService(ObjectMapper om) {
        this.om = om;
    }

    @PostConstruct
    public void load() {
        try {
            ClassPathResource res = new ClassPathResource("help/help-cards.json");
            try (InputStream is = res.getInputStream()) {
                HelpCardsFile file = om.readValue(is, HelpCardsFile.class);
                this.cards = (file != null && file.cards != null) ? file.cards : new ArrayList<>();
                this.byId = this.cards.stream()
                        .filter(c -> c != null && c.id != null)
                        .collect(Collectors.toMap(c -> c.id, c -> c, (a, b) -> a));
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to load help-cards.json", e);
        }
    }

    public List<String> categories() {
        return cards.stream()
                .map(c -> c.category)
                .filter(Objects::nonNull)
                .distinct()
                .sorted()
                .toList();
    }

    public List<HelpCard> list(String category, String q) {
        String cat = normalize(category);
        String qq = normalize(q);

        return cards.stream()
                .filter(c -> c != null)
                .filter(c -> cat.isBlank() || normalize(c.category).equals(cat))
                .filter(c -> qq.isBlank() || matches(c, qq))
                .limit(200)
                .toList();
    }

    public HelpCard get(String id) {
        return byId.get(id);
    }

    /**
     * ✅ 핵심: 직접 입력해도 카드가 나오게 추천 강화
     * - token 기반 부분 일치 점수
     * - 0개면 fallback 카드 반환
     */
    public List<HelpCard> recommend(String category, String message, int limit) {
        String cat = normalize(category);
        String msg = normalize(message);

        // 메시지가 비어있으면 fallback 먼저
        if (msg.isBlank()) {
            return fallback(cat, limit);
        }

        List<Scored> scored = new ArrayList<>();

        for (HelpCard c : cards) {
            if (c == null) continue;
            if (!cat.isBlank() && !normalize(c.category).equals(cat)) continue;

            int s = scoreTokens(c, msg);
            scored.add(new Scored(c, s));
        }

        List<HelpCard> res = scored.stream()
                .filter(x -> x.score > 0)
                .sorted((a, b) -> Integer.compare(b.score, a.score))
                .limit(limit)
                .map(x -> x.card)
                .toList();

        if (res.isEmpty()) {
            return fallback(cat, limit);
        }
        return res;
    }

    // ----------------------------
    // 내부 로직
    // ----------------------------

    private boolean matches(HelpCard c, String qq) {
        String t = normalize(c.title);
        if (!t.isBlank() && t.contains(qq)) return true;

        if (c.symptoms != null) {
            for (String s : c.symptoms) {
                if (normalize(s).contains(qq)) return true;
            }
        }
        if (c.tags != null) {
            for (String s : c.tags) {
                if (normalize(s).contains(qq)) return true;
            }
        }
        return false;
    }

    private List<HelpCard> fallback(String cat, int limit) {
        List<String> ids;
        if ("call".equals(cat)) {
            ids = List.of("call-010", "call-001", "call-007");
        } else if ("error".equals(cat)) {
            ids = List.of("err-001", "err-003", "err-008");
        } else { // camera or empty
            ids = List.of("cam-001", "cam-004", "cam-005");
        }

        List<HelpCard> res = new ArrayList<>();
        for (String id : ids) {
            HelpCard c = byId.get(id);
            if (c != null) res.add(c);
            if (res.size() >= limit) break;
        }

        // 혹시 fallback id가 JSON에 없으면, 같은 카테고리에서 앞쪽 카드라도 반환
        if (res.isEmpty()) {
            for (HelpCard c : cards) {
                if (c == null) continue;
                if (!cat.isBlank() && !normalize(c.category).equals(cat)) continue;
                res.add(c);
                if (res.size() >= limit) break;
            }
        }
        return res;
    }

    private int scoreTokens(HelpCard c, String msgNorm) {
        Set<String> msgTokens = tokenize(msgNorm);
        if (msgTokens.isEmpty()) return 0;

        int score = 0;

        // title (약)
        score += tokenHit(msgTokens, normalize(c.title)) * 2;

        // symptoms (강)
        if (c.symptoms != null) {
            for (String s : c.symptoms) score += tokenHit(msgTokens, normalize(s)) * 6;
        }

        // tags (중)
        if (c.tags != null) {
            for (String t : c.tags) score += tokenHit(msgTokens, normalize(t)) * 3;
        }

        // quickChecks/steps (약)
        if (c.quickChecks != null) {
            for (String q : c.quickChecks) score += tokenHit(msgTokens, normalize(q)) * 1;
        }
        if (c.steps != null) {
            for (var st : c.steps) {
                score += tokenHit(msgTokens, normalize(st.label)) * 1;
                score += tokenHit(msgTokens, normalize(st.detail)) * 1;
            }
        }

        return score;
    }

    private int tokenHit(Set<String> msgTokens, String textNorm) {
        if (textNorm == null || textNorm.isBlank()) return 0;
        Set<String> tks = tokenize(textNorm);
        int hit = 0;
        for (String tk : tks) {
            if (msgTokens.contains(tk)) hit++;
        }
        return hit;
    }

    private Set<String> tokenize(String norm) {
        if (norm == null || norm.isBlank()) return Set.of();
        String[] parts = norm.split(" ");
        Set<String> out = new HashSet<>();
        for (String p : parts) {
            if (p.length() >= 2) out.add(p);
        }
        return out;
    }

    private String normalize(String s) {
        if (s == null) return "";
        return s.toLowerCase()
                .replaceAll("[^a-z0-9가-힣\\s]", " ")
                .replaceAll("\\s+", " ")
                .trim();
    }

    private static class Scored {
        HelpCard card;
        int score;
        Scored(HelpCard c, int s) { this.card = c; this.score = s; }
    }
}
