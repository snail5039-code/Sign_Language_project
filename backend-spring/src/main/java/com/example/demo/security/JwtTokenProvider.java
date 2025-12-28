package com.example.demo.security;

import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Date;
import java.util.List;

import javax.crypto.SecretKey;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.stereotype.Component;

import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.security.Keys;


@Component
public class JwtTokenProvider {

	@Value("${jwt.secret}")
	private String secret;
	
	@Value("${jwt.expMinutes:60}")
	private int expMinutes;
	
	private SecretKey key() {
		return Keys.hmacShaKeyFor(secret.getBytes(StandardCharsets.UTF_8));
	}
	
	public String createToken(Integer memberId, String loginId) {
		Date now = new Date();
		Date exp = Date.from(Instant.now().plus(expMinutes, ChronoUnit.MINUTES));
		
		return Jwts.builder().subject(String.valueOf(memberId))
				.claim("loginId", loginId).claim("role", "ROLE_USER")
				.issuedAt(now).expiration(exp).signWith(key()).compact();
	}
	
	public Authentication toAuthentication(String token) {
		var claims = Jwts.parser().verifyWith(key()).build()
				.parseSignedClaims(token).getPayload();
		
		Integer memberId = Integer.valueOf(claims.getSubject());
		String role = (String) claims.get("role");
		
		return new UsernamePasswordAuthenticationToken(
                memberId,
                null,
                List.of(new SimpleGrantedAuthority(role))
        );
	}
}
