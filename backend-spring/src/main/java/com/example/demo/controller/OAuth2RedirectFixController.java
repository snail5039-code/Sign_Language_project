package com.example.demo.controller;

import java.io.IOException;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

@RestController
public class OAuth2RedirectFixController {

    @GetMapping("/oauth2/success")
    public void success(HttpServletRequest req, HttpServletResponse res) throws IOException {
        String q = req.getQueryString(); // accessToken=...
        String url = "http://localhost:5173/oauth2/success" + (q != null ? "?" + q : "");
        res.sendRedirect(url);
    }
}
