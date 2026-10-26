import cv2
import numpy as np
import os

#Task B: find every paragraph on a scanned paper and save each one as its own image, in reading order.
#Method: "projection" = count how many text pixels are in each row/column, then look for gaps between them.

INPUT_DIR = os.path.join("assets", "Converted Paper (8)")   #Folder with the 8 paper images.
OUTPUT_DIR = "output_paragraphs"                             #Folder where cropped paragraphs are saved.

MIN_COLUMN_GAP_WIDTH = 20            #A column gap must be at least this many pixels wide to count.
MIN_LINE_GAP_HEIGHT = 3              #Smallest gap (in pixels) allowed between two lines of text.
MIN_COLUMN_WIDTH_RATIO = 0.05        #A column must be at least this fraction of the page width.
PARAGRAPH_GAP_MULTIPLIER = 2.2       #A gap bigger than (median line gap x this) = new paragraph.
CROP_PADDING = 8                     #Extra pixels of padding added around each cropped paragraph.
TABLE_LINE_RATIO = 0.5               #A row with an unbroken ink run this long (vs block width) = a table border.
FIGURE_DENSITY_THRESHOLD = 0.25      #A block with this much of its area filled with ink = likely a photo/figure.


def load_and_binarize(image_path):
    image = cv2.imread(image_path)                                    #Read the paper image.
    if image is None:
        raise IOError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                     #Convert to grayscale.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  #Text=white, background=black.
    return image, binary


def find_zero_runs(profile, min_width):
    #Find stretches of consecutive zeros (blank rows/columns) at least "min_width" long.
    runs = []
    start = None
    for i, value in enumerate(profile):
        if value == 0 and start is None:            #Start of a new blank run.
            start = i
        elif value != 0 and start is not None:       #End of that blank run.
            if i - start >= min_width:
                runs.append([start, i - 1])
            start = None
    if start is not None and len(profile) - start >= min_width:   #Handle a blank run that reaches the edge.
        runs.append([start, len(profile) - 1])
    return runs


def find_column_boundaries(binary):
    #Vertical projection: sum text pixels down each column to find gaps between columns.
    height, width = binary.shape
    col_profile = np.sum(binary > 0, axis=0)                       #Text-pixel count per column.

    zero_runs = find_zero_runs(col_profile, MIN_COLUMN_GAP_WIDTH)  #Wide blank column gaps = column separators.

    cut_points = [0]                                                #Turn each gap into a page "cut point".
    for run in zero_runs:
        cut_points.append((run[0] + run[1]) // 2)
    cut_points.append(width)
    cut_points = sorted(set(cut_points))

    columns = []
    for i in range(len(cut_points) - 1):                            #Build column regions between cut points.
        x_start, x_end = cut_points[i], cut_points[i + 1]
        if x_end - x_start < width * MIN_COLUMN_WIDTH_RATIO:
            continue                                                #Too thin - just a margin, skip it.
        if np.sum(col_profile[x_start:x_end]) == 0:
            continue                                                #No text at all - just a margin, skip it.
        columns.append([x_start, x_end])

    return columns


def find_line_bands(column_binary):
    #Horizontal projection: sum text pixels across each row to find individual lines of text.
    row_profile = np.sum(column_binary > 0, axis=1)
    has_text = row_profile > 0

    bands = []
    start = None
    for y, present in enumerate(has_text):
        if present and start is None:            #Start of a line of text.
            start = y
        elif not present and start is not None:   #End of that line of text.
            bands.append([start, y - 1])
            start = None
    if start is not None:
        bands.append([start, len(has_text) - 1])

    return bands


def group_lines_into_paragraphs(line_bands):
    #Merge consecutive lines into one paragraph, based on the gap between them.
    if not line_bands:
        return []
    if len(line_bands) == 1:
        return [line_bands[0]]

    gaps = [line_bands[i + 1][0] - line_bands[i][1] for i in range(len(line_bands) - 1)]  #Gap after each line.
    median_gap = np.median(gaps)                                   #Typical (normal) line spacing.
    threshold = max(median_gap * PARAGRAPH_GAP_MULTIPLIER, MIN_LINE_GAP_HEIGHT * 3)  #Bigger gap = new paragraph.

    paragraphs = []
    para_start = line_bands[0][0]
    para_end = line_bands[0][1]
    for i in range(1, len(line_bands)):
        gap = line_bands[i][0] - line_bands[i - 1][1]
        if gap > threshold:                          #Gap too big - close this paragraph, start a new one.
            paragraphs.append([para_start, para_end])
            para_start = line_bands[i][0]
        para_end = line_bands[i][1]
    paragraphs.append([para_start, para_end])         #Add the last paragraph.

    return paragraphs


def find_row_sections(binary, min_gap_height=30):
    #Full-width horizontal projection: splits the page where a row is completely blank across its whole width.
    #Needed for pages that mix layouts, e.g. a full-width table sitting above double-column text.
    height, width = binary.shape
    row_profile = np.sum(binary > 0, axis=1)

    zero_runs = find_zero_runs(row_profile, min_gap_height)         #Fully blank rows = a layout change.

    cut_points = [0]
    for run in zero_runs:
        cut_points.append((run[0] + run[1]) // 2)
    cut_points.append(height)
    cut_points = sorted(set(cut_points))

    sections = []
    for i in range(len(cut_points) - 1):
        y_start, y_end = cut_points[i], cut_points[i + 1]
        if np.sum(row_profile[y_start:y_end]) == 0:
            continue                                                #Blank section (margin) - skip it.
        sections.append([y_start, y_end])

    return sections


def is_table_block(block_binary, line_ratio_threshold=TABLE_LINE_RATIO):
    #A table has ruled border lines - long unbroken runs of ink spanning most of the block's width.
    height, width = block_binary.shape
    for row in block_binary:                          #Loop through every row in the block.
        row_is_ink = row > 0
        longest_run = 0
        current_run = 0
        for pixel_is_ink in row_is_ink:                #Loop through the row, pixel by pixel.
            if pixel_is_ink:
                current_run += 1                       #Extend the current unbroken run of ink.
                longest_run = max(longest_run, current_run)
            else:
                current_run = 0                        #Ink run broken by a gap - reset the counter.
        if longest_run / width >= line_ratio_threshold:  #This row has a border-line-length run of ink.
            return True
    return False


def is_figure_block(block_binary, density_threshold=FIGURE_DENSITY_THRESHOLD):
    #A photo/figure has no blank rows breaking it into separate lines (it is one continuous block),
    height, width = block_binary.shape
    row_profile = np.sum(block_binary > 0, axis=1)          #Text-pixel count per row.
    has_ink = row_profile > 0
    number_of_bands = 0                                      #Count how many separate line-bands this block has.
    previous_had_ink = False
    for row_has_ink in has_ink:                              #Loop through every row.
        if row_has_ink and not previous_had_ink:
            number_of_bands += 1                              #A new band starts here.
        previous_had_ink = row_has_ink

    ink_density = np.sum(block_binary > 0) / (height * width)  #Fraction of the block that is "ink".
    return number_of_bands <= 1 and ink_density >= density_threshold  #One solid block + lots of ink = a photo.


def extract_paragraphs_from_image(image_path, output_dir, paper_name):
    image, binary = load_and_binarize(image_path)                   #Step 1: binarize the page.
    height, width = binary.shape

    os.makedirs(output_dir, exist_ok=True)
    paragraph_index = 1

    row_sections = find_row_sections(binary)                        #Step 2: split page into horizontal sections.

    for section in row_sections:                                    #Loop through sections, top to bottom.
        y_start, y_end = section[0], section[1]
        section_binary = binary[y_start:y_end, :]
        columns = find_column_boundaries(section_binary)            #Step 3: find columns in this section.

        for column in columns:                                      #Loop through columns, left to right.
            x_start, x_end = column[0], column[1]
            column_binary = section_binary[:, x_start:x_end]
            line_bands = find_line_bands(column_binary)              #Step 4: find lines of text in this column.
            paragraphs = group_lines_into_paragraphs(line_bands)     #Step 5: group lines into paragraphs.

            for para in paragraphs:                                  #Loop through paragraphs, top to bottom.
                local_y_start, local_y_end = para[0], para[1]
                top = max(y_start + local_y_start - CROP_PADDING, 0)      #Convert back to full-page coordinates.
                bottom = min(y_start + local_y_end + CROP_PADDING, height)
                left = max(x_start - CROP_PADDING, 0)
                right = min(x_end + CROP_PADDING, width)

                block_binary = binary[top:bottom, left:right]        #Step 6: get this block's binary pixels.
                if is_table_block(block_binary):                     #Step 7: skip it if it looks like a table.
                    continue
                if is_figure_block(block_binary):                    #Step 8: skip it if it looks like a photo/figure.
                    continue

                crop = image[top:bottom, left:right]                 #Step 9: crop out this paragraph.
                out_path = os.path.join(output_dir, f"{paper_name}_paragraph_{paragraph_index}.png")
                cv2.imwrite(out_path, crop)                          #Save it as its own file.
                paragraph_index += 1

    return paragraph_index - 1


def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"[ERROR] Input folder not found: {INPUT_DIR}")
        return

    image_files = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".png"))  #All 8 paper images.

    for image_file in image_files:                                  #Loop through every paper.
        paper_name = os.path.splitext(image_file)[0]
        image_path = os.path.join(INPUT_DIR, image_file)
        out_dir = os.path.join(OUTPUT_DIR, paper_name)

        count = extract_paragraphs_from_image(image_path, out_dir, paper_name)   #Run the pipeline on this paper.
        print(f"{image_file}: extracted {count} paragraph(s) -> {out_dir}")


if __name__ == "__main__":
    main()
