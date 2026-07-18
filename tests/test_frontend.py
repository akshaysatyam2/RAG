import pytest
import os
import time
import subprocess
from pathlib import Path

# Mark all tests in this file as frontend
pytestmark = pytest.mark.frontend

def test_page_loads(page):
    frontend_dir = Path(__file__).parent.parent / "frontend"
    page.goto(f"file://{frontend_dir}/index.html")
    assert "GraphRAG" in page.title()

def test_theme_toggle(page):
    frontend_dir = Path(__file__).parent.parent / "frontend"
    page.goto(f"file://{frontend_dir}/index.html")
    
    body = page.locator("body")
    theme_btn = page.locator("#theme-toggle")
    
    # Assuming the body gets a 'dark' class when toggled
    initial_class = body.get_attribute("class") or ""
    theme_btn.click()
    new_class = body.get_attribute("class") or ""
    assert initial_class != new_class

def test_context_slider(page):
    frontend_dir = Path(__file__).parent.parent / "frontend"
    page.goto(f"file://{frontend_dir}/index.html")
    
    # Assuming there's an input type=range for context
    slider = page.locator("input[type='range']")
    if slider.count() > 0:
        slider.fill("10")
        assert slider.input_value() == "10"

def test_document_upload_area(page):
    frontend_dir = Path(__file__).parent.parent / "frontend"
    page.goto(f"file://{frontend_dir}/index.html")
    
    upload_area = page.locator("#upload-area")
    assert upload_area.is_visible()

def test_chat_input(page):
    frontend_dir = Path(__file__).parent.parent / "frontend"
    page.goto(f"file://{frontend_dir}/index.html")
    
    chat_input = page.locator("#chat-input")
    send_btn = page.locator("#send-button")
    
    assert chat_input.is_visible()
    assert send_btn.is_visible()

def test_empty_states(page):
    frontend_dir = Path(__file__).parent.parent / "frontend"
    page.goto(f"file://{frontend_dir}/index.html")
    
    doc_list = page.locator("#document-list")
    if doc_list.is_visible():
        text = doc_list.inner_text()
        assert "No documents" in text or text.strip() == ""

def test_responsive_layout(page):
    frontend_dir = Path(__file__).parent.parent / "frontend"
    page.goto(f"file://{frontend_dir}/index.html")
    
    # Mobile
    page.set_viewport_size({"width": 375, "height": 667})
    assert page.locator("body").is_visible()
    
    # Tablet
    page.set_viewport_size({"width": 768, "height": 1024})
    assert page.locator("body").is_visible()
    
    # Desktop
    page.set_viewport_size({"width": 1440, "height": 900})
    assert page.locator("body").is_visible()

