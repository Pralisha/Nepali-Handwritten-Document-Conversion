def perform_ocr(cropped_word_images, processor, model, device, tokenizer):
    compiled_lines = {}
    word_mappings = []  # List to hold tuples of (cropped image, predicted word)
    
    for img, img_file in cropped_word_images:
        pixel_values = processor(images=img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        output_ids = model.generate(pixel_values)
        predicted_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract line number from the image file name
        line_number = int(img_file.split('_')[0])

        # Update compiled lines for displaying full text
        if line_number not in compiled_lines:
            compiled_lines[line_number] = []
        compiled_lines[line_number].append(predicted_text)

        # Append a tuple of the cropped image and its predicted word
        word_mappings.append((img, predicted_text,img_file))

    # Prepare final compiled text
    final_output_lines = []
    for line_number in sorted(compiled_lines.keys()):
        final_output_lines.append(' '.join(compiled_lines[line_number]))

    final_output = '\n'.join(final_output_lines)

    return final_output, word_mappings  # Return both compiled text and word mappings




