import openai
import os
import base64
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

# It's recommended to create a .env file and add your OpenAI API key to it
# OPENAI_API_KEY=your_key_here
load_dotenv()

client = openai.OpenAI()

class ScenarioPair(BaseModel):
    success: str = Field(description="The achievement or success scene.")
    work: str = Field(description="The corresponding hard work or habit-building scene.")
    top_text: str = Field(description="The text overlay for the top panel.")
    bottom_text: str = Field(description="The text overlay for the bottom panel.")

class ComicScenarioPairs(BaseModel):
    scenarios: List[ScenarioPair] = Field(
        description="A list of exactly 4 pairs of success/work scenarios for a habit-themed comic strip with varied messaging.",
        min_length=4,
        max_length=4
    )

def create_final_prompt(scenario: ScenarioPair, slide_number: int) -> str:
    """
    Creates a detailed, final prompt for image generation by injecting a scenario pair into a style template.
    """
    engagement_instruction = ""
    if slide_number == 1:  # First slide - mandatory edgy elements
        engagement_instruction = """<background_details>MANDATORY: Include provocative, funny, or attention-grabbing background elements that boost engagement. These can be edgy, non-tasteful, or controversial - whatever stops the scroll. NO POLITICAL CONTENT. Examples: provocative posters, funny signs, ironic objects, visual jokes that complement the hook message.</background_details>"""
    elif slide_number == 3:  # Third slide - subtle provocative elements
        engagement_instruction = """<background_details>SUBTLE PROVOCATIVE: Add small, hidden provocative elements in the background that viewers might notice on second look. Could include subtle sexual imagery, suggestive objects, or provocative details that don't overwhelm the main scene but add engagement. NO POLITICAL CONTENT. Keep it tasteful but noticeable.</background_details>"""
    else:  # Other slides - clean focus
        engagement_instruction = """<background_details>CLEAN FOCUS: Keep background simple and focused on the main message without distracting elements.</background_details>"""

    return f"""<instructions>
<format>
- Single image, exactly 1024x1536 pixels
- Vertically stacked 2-panel comic
- Panels MUST blend seamlessly with NO dividing line or border between them
- Artwork MUST extend to all four edges (full-bleed, no margins/padding)
</format>

<style_guide>
<consistency>ABSOLUTE CRITICAL: This comic MUST be VISUALLY IDENTICAL in style to all other comics in the series. NO VARIATION ALLOWED. Use the EXACT same art style, character design, color palette, and rendering approach across ALL 4 comics.</consistency>

<mandatory_style>LOCKED CARTOON STYLE: Modern, vibrant, high-quality cartoon illustration style with these EXACT characteristics:
- Bright, saturated colors with warm lighting
- Smooth, polished cartoon rendering (like Pixar/Disney style)
- Characters with exaggerated but appealing cartoon features
- Clean, rounded character designs with soft edges
- Vibrant backgrounds with rich color palettes
- NO realistic or semi-realistic rendering
- NO muted or desaturated colors
- NO photorealistic elements
- Consistent cartoon proportions and facial features across ALL characters
</mandatory_style>

<characters>
- IDENTICAL character design across ALL panels and ALL comics
- Cartoon-style with exaggerated but appealing features
- Same facial structure, eye shape, nose, and mouth design
- Consistent body proportions and clothing style
- Bright, vibrant character coloring
- Expressive cartoon faces with clear emotions
</characters>

<visual_treatment>
- Bright, vibrant color palette with warm lighting
- Smooth cartoon shading with soft gradients
- Rich, saturated colors throughout
- Dynamic but cartoon-appropriate lighting
- NO realistic textures or photographic elements
- Consistent cartoon rendering quality across all elements
</visual_treatment>

<content_guidelines>
- NO POLITICAL CONTENT of any kind
</content_guidelines>

<text_formatting>
- Text overlaid directly on artwork in clean, legible, lowercase, sans-serif font
- NO shapes, boxes, cards, speech bubbles, or thought bubbles around text
- Text placed directly on the image surface
- ABSOLUTE CRITICAL: Text must be positioned at the BOTTOM CENTER of each panel with MANDATORY margins from all edges
- ABSOLUTE CRITICAL: Text MUST be completely visible within panel boundaries - NO text can be cut off or extend beyond panel edges
- ABSOLUTE CRITICAL: Leave at least 40-60 pixels margin from the bottom edge of each panel
- ABSOLUTE CRITICAL: Leave at least 30-40 pixels margin from left and right edges of each panel
- Text size must be IDENTICAL across all panels and all comics - use a large, bold, consistent font size
- Text should be approximately 24-28pt equivalent size for readability (smaller if needed to fit)
- Text must be horizontally centered and have adequate margin from ALL panel edges
- Text color should be white or light colored for maximum readability
- If text is too long, reduce font size slightly to ensure it fits completely within margins
</text_formatting>

<engagement_strategy>
Slide {slide_number}: {"MANDATORY edgy elements for hook (NO POLITICS)" if slide_number == 1 else "SUBTLE provocative elements (NO POLITICS)" if slide_number == 3 else "CLEAN focus on message"}
</engagement_strategy>
</style_guide>

<content>
<top_panel>
<scene>{scenario.success}</scene>
{engagement_instruction}
<text_position>Bottom center of the top panel, with MANDATORY 40-60px margin from bottom edge and 30-40px margin from left/right edges</text_position>
<text_size>Large, bold, consistent sizing (24-28pt equivalent) - MUST fit completely within panel margins</text_size>
<text>{scenario.top_text}</text>
</top_panel>

<bottom_panel>
<scene>Same character from top panel performing the underlying habit: {scenario.work}. Character could be taking a picture of their progress (reflecting 'proofs' app concept).</scene>
{engagement_instruction}
<text_position>Bottom center of the bottom panel, with MANDATORY 40-60px margin from bottom edge and 30-40px margin from left/right edges</text_position>
<text_size>Large, bold, consistent sizing (24-28pt equivalent) - MUST match top panel exactly and fit completely within panel margins</text_size>
<text>{scenario.bottom_text}</text>
</bottom_panel>
</content>
</instructions>"""

def generate_scenarios():
    """
    Generates 4 pairs of scenarios for the comic strips using a language model.
    """
    print("Generating comic scenarios...")
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """<role>
You are a creative assistant for "proofs," a social habit-building app where users prove they've done a habit by taking a picture.
</role>

<task>
Generate EXACTLY 4 distinct scenario pairs for a 2-panel comic slideshow.
These will be shown in sequence, so the first one MUST be controversial/attention-grabbing to hook users.
</task>

<requirements>
<first_comic>ABSOLUTELY CRITICAL: The first comic MUST be INSANELY compelling and controversial. This single slide determines if users see slides 2-4. It must:
- Challenge deeply held beliefs about success
- Make users feel called out or defensive
- Use provocative language that stops scrolling
- Create an emotional reaction (anger, curiosity, or validation)
- Be impossible to ignore or scroll past
- Make bold claims that most people disagree with</first_comic>

<scenario_structure>
Each pair must consist of:
1. "success" scene: Character enjoying the results of a consistent habit
2. "work" scene: Same character performing the specific habit that led to that success
3. "top_text": Text for the top panel (MUST be insanely provocative/controversial for first comic)
4. "bottom_text": Text for the bottom panel (the habit/work message)
</scenario_structure>

<high_converting_hooks>
PROVEN HIGH-CONVERTING FIRST COMIC EXAMPLES:
- "everyone lies about their success" / "i document the truth"
- "your excuses are pathetic" / "my habits are powerful"
- "most people are fake" / "i prove everything"
- "success isn't for everyone" / "it's for people like me"
- "you're probably not ready" / "i was born ready"
- "stop lying to yourself" / "start proving yourself"
- "your friends hold you back" / "i leave them behind"
- "comfort kills dreams" / "discomfort builds them"
- "you talk, i do" / "you wish, i work"
- "mediocrity is a choice" / "excellence is my habit"
- "you make excuses" / "i make progress"
- "lazy people hate this" / "successful people love it"
- "this isn't for quitters" / "this is for winners"
- "you scroll, i build" / "you watch, i achieve"
- "your potential is wasted" / "mine is proven daily"

OTHER SLIDES (less controversial but still engaging):
- "they said impossible" / "i said watch me"
- "overnight success took years" / "every single day"
- "talent is overrated" / "consistency is underrated"
- "results don't lie" / "neither do my habits"
</high_converting_hooks>

<constraints>
- Scenarios must be strictly habit-related and visually distinct
- First comic should be INSANELY attention-grabbing and controversial
- All text should be lowercase
- Focus on habits that can be "proved" with photos
- Make statements that challenge conventional wisdom and trigger emotional responses
- The first slide must be impossible to scroll past
- Use direct, confrontational language that makes people feel called out
</constraints>
</requirements>

<output_format>
JSON object conforming to the Pydantic schema, containing exactly 4 scenario pairs.
</output_format>"""
                },
                {
                    "role": "user",
                    "content": "Generate 4 habit-based achievement scenario pairs for our app 'proofs', with the first one being INSANELY controversial and attention-grabbing using the high-converting hook examples provided."
                }
            ],
            response_format=ComicScenarioPairs,
        )

        message = completion.choices[0].message
        if message.refusal:
            print(f"Scenario generation refused: {message.refusal}")
            return []
        
        scenarios = message.parsed.scenarios
        print("Successfully generated scenarios.")
        return scenarios
    except Exception as e:
        print(f"Error generating scenarios: {e}")
        return []

def generate_image(prompt, index, output_folder):
    """
    Generates a single image based on a prompt.
    """
    print(f"Generating image {index + 1}...")
    try:
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            quality="high",
            size="1024x1536"
        )
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        
        filename = f"comic_{index + 1}.png"
        filepath = os.path.join(output_folder, filename)
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        print(f"Successfully saved image {index + 1} as {filepath}")
        return filepath
    except Exception as e:
        print(f"Error generating image {index + 1}: {e}")
        return None

def main():
    """
    Main function to generate comic slideshow.
    """
    # Create timestamped folder for this generation
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = f"generation_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    print(f"Created output folder: {output_folder}")
    
    scenarios = generate_scenarios()
    
    if not scenarios:
        print("Could not generate scenarios. Exiting.")
        return

    final_prompts = []
    for i, scenario in enumerate(scenarios):
        slide_num = i + 1  # Convert to 1-based numbering
        prompt = create_final_prompt(scenario, slide_number=slide_num)
        final_prompts.append(prompt)
    
    print(f"Generated {len(final_prompts)} prompts for parallel processing...")
    print("Engagement strategy: Slide 1 (edgy), Slide 2 (clean), Slide 3 (subtle provocative), Slide 4 (clean)")

    # Use ThreadPoolExecutor with max_workers=4 to ensure all 4 images process in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        print("Starting parallel image generation...")
        futures = [executor.submit(generate_image, prompt, i, output_folder) for i, prompt in enumerate(final_prompts)]
        
        # Wait for all futures to complete
        results = []
        for i, future in enumerate(futures):
            try:
                result = future.result()
                results.append(result)
                print(f"Completed image {i + 1}")
            except Exception as e:
                print(f"Failed to generate image {i + 1}: {e}")
                results.append(None)

    successful_images = [r for r in results if r]
    print(f"\nSlideshow generation complete! Generated {len(successful_images)} out of 4 images.")
    print(f"Images saved in folder: {output_folder}")

if __name__ == "__main__":
    main() 