#!/usr/bin/env python3
"""
Simple Frontend Test Runner for Study Assistant

This script runs basic validation tests on the frontend files without requiring
external dependencies like Selenium or complex testing frameworks.

Usage:
    python tests/run_frontend_tests.py
"""

import os
import sys
import json
import re
from pathlib import Path

class FrontendTestRunner:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.tests_passed = 0
        self.tests_failed = 0
        self.tests_total = 0
    
    def run_all_tests(self):
        """Run all frontend validation tests"""
        print("🧪 Study Assistant Frontend Test Runner")
        print("=" * 50)
        
        # Test HTML structure
        self.test_html_structure()
        
        # Test CSS validity
        self.test_css_validity()
        
        # Test JavaScript syntax
        self.test_javascript_syntax()
        
        # Test file accessibility
        self.test_file_accessibility()
        
        # Test responsive design elements
        self.test_responsive_design()
        
        # Test accessibility features
        self.test_accessibility_features()
        
        # Print summary
        self.print_summary()
    
    def test_html_structure(self):
        """Test HTML template structure and required elements"""
        print("\n📄 Testing HTML Structure...")
        
        html_file = self.project_root / "templates" / "index.html"
        
        if not html_file.exists():
            self.fail("HTML template file not found")
            return
        
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Required elements
        required_elements = [
            ('DOCTYPE declaration', r'<!DOCTYPE html>'),
            ('Viewport meta tag', r'<meta name="viewport"'),
            ('Title tag', r'<title>.*Study Assistant.*</title>'),
            ('Upload area', r'id="uploadArea"'),
            ('File input', r'id="fileInput"'),
            ('Question input', r'id="questionInput"'),
            ('Ask button', r'id="askButton"'),
            ('Documents list', r'id="documentsList"'),
            ('Answer section', r'id="answerSection"'),
            ('CSS link', r'href="/static/style.css"'),
            ('JavaScript link', r'src="/static/app.js"')
        ]
        
        for name, pattern in required_elements:
            if re.search(pattern, html_content, re.IGNORECASE):
                self.pass_test(f"✓ {name} found")
            else:
                self.fail_test(f"✗ {name} missing")
        
        # Check for proper HTML5 structure
        html5_elements = ['<header>', '<main>', '<section>', '<footer>']
        for element in html5_elements:
            if element in html_content:
                self.pass_test(f"✓ HTML5 element {element} found")
            else:
                self.fail_test(f"✗ HTML5 element {element} missing")
    
    def test_css_validity(self):
        """Test CSS file for basic validity and required styles"""
        print("\n🎨 Testing CSS Validity...")
        
        css_file = self.project_root / "static" / "style.css"
        
        if not css_file.exists():
            self.fail_test("CSS file not found")
            return
        
        with open(css_file, 'r', encoding='utf-8') as f:
            css_content = f.read()
        
        # Required CSS classes/selectors
        required_styles = [
            '.container',
            '.upload-area',
            '.question-form',
            '.document-item',
            '.answer-section',
            'header',
            'main',
            'section'
        ]
        
        for style in required_styles:
            if style in css_content:
                self.pass_test(f"✓ CSS selector {style} found")
            else:
                self.fail_test(f"✗ CSS selector {style} missing")
        
        # Check for responsive design
        if '@media' in css_content:
            self.pass_test("✓ Responsive design media queries found")
        else:
            self.fail_test("✗ No responsive design media queries found")
        
        # Check for CSS variables
        if ':root' in css_content and '--' in css_content:
            self.pass_test("✓ CSS custom properties (variables) found")
        else:
            self.fail_test("✗ CSS custom properties not found")
        
        # Basic syntax check (count braces)
        open_braces = css_content.count('{')
        close_braces = css_content.count('}')
        if open_braces == close_braces:
            self.pass_test("✓ CSS brace matching is correct")
        else:
            self.fail_test(f"✗ CSS brace mismatch: {open_braces} open, {close_braces} close")
    
    def test_javascript_syntax(self):
        """Test JavaScript file for basic syntax and required functions"""
        print("\n⚡ Testing JavaScript Syntax...")
        
        js_file = self.project_root / "static" / "app.js"
        
        if not js_file.exists():
            self.fail_test("JavaScript file not found")
            return
        
        with open(js_file, 'r', encoding='utf-8') as f:
            js_content = f.read()
        
        # Required classes and functions
        required_elements = [
            ('StudyAssistant class', r'class StudyAssistant'),
            ('Constructor', r'constructor\(\)'),
            ('Upload file method', r'async uploadFile'),
            ('Ask question method', r'async askQuestion'),
            ('Load documents method', r'async loadDocuments'),
            ('Delete document method', r'async deleteDocument'),
            ('Drag over handler', r'handleDragOver'),
            ('Drag leave handler', r'handleDragLeave'),
            ('Drop handler', r'handleDrop'),
            ('File select handler', r'handleFileSelect')
        ]
        
        for name, pattern in required_elements:
            if re.search(pattern, js_content):
                self.pass_test(f"✓ {name} found")
            else:
                self.fail_test(f"✗ {name} missing")
        
        # Check for modern JavaScript features
        modern_features = [
            ('Arrow functions', r'=>'),
            ('Async/await', r'async|await'),
            ('Template literals', r'`.*\$\{.*\}.*`'),
            ('Destructuring', r'\{.*\}.*='),
            ('Fetch API', r'fetch\(')
        ]
        
        for name, pattern in modern_features:
            if re.search(pattern, js_content):
                self.pass_test(f"✓ {name} used")
        
        # Basic syntax checks
        open_braces = js_content.count('{')
        close_braces = js_content.count('}')
        if open_braces == close_braces:
            self.pass_test("✓ JavaScript brace matching is correct")
        else:
            self.fail_test(f"✗ JavaScript brace mismatch: {open_braces} open, {close_braces} close")
        
        open_parens = js_content.count('(')
        close_parens = js_content.count(')')
        if open_parens == close_parens:
            self.pass_test("✓ JavaScript parentheses matching is correct")
        else:
            self.fail_test(f"✗ JavaScript parentheses mismatch: {open_parens} open, {close_parens} close")
    
    def test_file_accessibility(self):
        """Test that all required files exist and are accessible"""
        print("\n📁 Testing File Accessibility...")
        
        required_files = [
            ("HTML template", "templates/index.html"),
            ("CSS stylesheet", "static/style.css"),
            ("JavaScript file", "static/app.js")
        ]
        
        for name, file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                # Check file size
                size = full_path.stat().st_size
                if size > 0:
                    self.pass_test(f"✓ {name} exists and has content ({size} bytes)")
                else:
                    self.fail_test(f"✗ {name} exists but is empty")
            else:
                self.fail_test(f"✗ {name} not found at {file_path}")
    
    def test_responsive_design(self):
        """Test responsive design implementation"""
        print("\n📱 Testing Responsive Design...")
        
        css_file = self.project_root / "static" / "style.css"
        html_file = self.project_root / "templates" / "index.html"
        
        if not css_file.exists() or not html_file.exists():
            self.fail_test("Required files not found for responsive design test")
            return
        
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        with open(css_file, 'r', encoding='utf-8') as f:
            css_content = f.read()
        
        # Check viewport meta tag
        if 'name="viewport"' in html_content and 'width=device-width' in html_content:
            self.pass_test("✓ Viewport meta tag configured correctly")
        else:
            self.fail_test("✗ Viewport meta tag missing or incorrect")
        
        # Check for media queries
        media_queries = re.findall(r'@media[^{]+\{', css_content)
        if media_queries:
            self.pass_test(f"✓ {len(media_queries)} media queries found")
            
            # Check for common breakpoints
            breakpoints = ['768px', '480px', '1024px']
            for bp in breakpoints:
                if bp in css_content:
                    self.pass_test(f"✓ Breakpoint {bp} found")
        else:
            self.fail_test("✗ No media queries found")
        
        # Check for flexible layouts
        if 'flex' in css_content or 'grid' in css_content:
            self.pass_test("✓ Modern layout methods (flexbox/grid) used")
        else:
            self.fail_test("✗ No modern layout methods found")
    
    def test_accessibility_features(self):
        """Test accessibility features"""
        print("\n♿ Testing Accessibility Features...")
        
        html_file = self.project_root / "templates" / "index.html"
        
        if not html_file.exists():
            self.fail_test("HTML file not found for accessibility test")
            return
        
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Check for accessibility attributes
        accessibility_features = [
            ('Alt attributes', r'alt='),
            ('ARIA labels', r'aria-'),
            ('Label elements', r'<label'),
            ('Semantic HTML', r'<(header|main|section|nav|footer)'),
            ('Language attribute', r'<html[^>]*lang=')
        ]
        
        for name, pattern in accessibility_features:
            if re.search(pattern, html_content, re.IGNORECASE):
                self.pass_test(f"✓ {name} found")
            else:
                self.fail_test(f"✗ {name} missing")
        
        # Check for proper heading hierarchy
        headings = re.findall(r'<h([1-6])', html_content, re.IGNORECASE)
        if headings:
            heading_levels = [int(h) for h in headings]
            if 1 in heading_levels:
                self.pass_test("✓ H1 heading found")
            else:
                self.fail_test("✗ No H1 heading found")
            
            # Check for logical heading order
            if heading_levels == sorted(heading_levels):
                self.pass_test("✓ Heading hierarchy is logical")
            else:
                self.fail_test("✗ Heading hierarchy may be incorrect")
    
    def pass_test(self, message):
        """Record a passed test"""
        print(f"  {message}")
        self.tests_passed += 1
        self.tests_total += 1
    
    def fail_test(self, message):
        """Record a failed test"""
        print(f"  {message}")
        self.tests_failed += 1
        self.tests_total += 1
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        print(f"Total Tests: {self.tests_total}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        
        if self.tests_total > 0:
            success_rate = (self.tests_passed / self.tests_total) * 100
            print(f"Success Rate: {success_rate:.1f}%")
        
        if self.tests_failed == 0:
            print("\n🎉 All tests passed! Frontend is ready.")
        else:
            print(f"\n⚠️  {self.tests_failed} test(s) failed. Please review the issues above.")
        
        print("\nNext steps:")
        print("1. Open tests/frontend_tests.html in your browser for interactive tests")
        print("2. Run the Study Assistant server and test manually")
        print("3. Use pytest tests/test_frontend_integration.py for API integration tests")


def main():
    """Main function to run all tests"""
    runner = FrontendTestRunner()
    runner.run_all_tests()
    
    # Return exit code based on test results
    return 0 if runner.tests_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())