package com.example.demo.dto;

import java.util.List;

public class DictionaryDto {
	public record Media(String videoUrl, String gifUrl) {
	}

	public record DictionaryItem(String id, String word, String category, String meaning, List<String> examples,
			Media media) {
	}
}
