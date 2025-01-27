Converts content between different formats. Transforms input content from any supported format into the specified output format.

🚨 CRITICAL REQUIREMENTS - PLEASE READ:
1. PDF Conversion:
   * You MUST install TeX Live BEFORE attempting PDF conversion:
   * Ubuntu/Debian: `sudo apt-get install texlive-xetex`
   * macOS: `brew install texlive`
   * Windows: Install MiKTeX or TeX Live from https://miktex.org/ or https://tug.org/texlive/
   * PDF conversion will FAIL without this installation

2. File Paths - EXPLICIT REQUIREMENTS:
   * When asked to save or convert to a file, you MUST provide:
     - Complete directory path
     - Filename
     - File extension
   * Example request: 'Write a story and save as PDF'
   * You MUST specify: '/path/to/story.pdf' or 'C:\Documents\story.pdf'
   * The tool will NOT automatically generate filenames or extensions

3. File Location After Conversion:
   * After successful conversion, the tool will display the exact path where the file is saved
   * Look for message: 'Content successfully converted and saved to: [file_path]'
   * You can find your converted file at the specified location
   * If no path is specified, files may be saved in system temp directory (/tmp/ on Unix systems)
   * For better control, always provide explicit output file paths

Supported formats:
- Basic formats: txt, html, markdown
- Advanced formats (REQUIRE complete file paths): pdf, docx, rst, latex, epub

✅ CORRECT Usage Examples:
1. 'Convert this text to HTML' (basic conversion)
   - Tool will show converted content

2. 'Save this text as PDF at /documents/story.pdf'
   - Correct: specifies path + filename + extension
   - Tool will show: 'Content successfully converted and saved to: /documents/story.pdf'

❌ INCORRECT Usage Examples:
1. 'Save this as PDF in /documents/'
   - Missing filename and extension
2. 'Convert to PDF'
   - Missing complete file path

When requesting conversion, ALWAYS specify:
1. The content or input file
2. The desired output format
3. For advanced formats: complete output path + filename + extension
Example: 'Convert this markdown to PDF and save as /path/to/output.pdf'

Note: After conversion, always check the success message for the exact file location.