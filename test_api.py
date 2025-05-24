#!/usr/bin/env python3
"""
Test script for NeMo ASR API
"""

import requests
import numpy as np
import soundfile as sf
import tempfile
import os
import time
import argparse

def generate_test_audio(duration=8, sample_rate=16000, filename="test_audio.wav"):
    """Generate a test audio file"""
    # Generate sine wave with some noise for testing
    t = np.linspace(0, duration, duration * sample_rate, False)
    frequency = 440  # A4 note
    audio = 0.3 * np.sin(2 * np.pi * frequency * t) + 0.1 * np.random.randn(len(t))
    
    # Save to file
    sf.write(filename, audio, sample_rate)
    print(f"Generated test audio: {filename} ({duration}s, {sample_rate}Hz)")
    return filename

def test_health_check(base_url):
    """Test health check endpoints"""
    print("\n=== Testing Health Check ===")
    
    try:
        # Test root endpoint
        response = requests.get(f"{base_url}/")
        print(f"Root endpoint: {response.status_code} - {response.json()}")
        
        # Test health endpoint
        response = requests.get(f"{base_url}/health")
        print(f"Health endpoint: {response.status_code} - {response.json()}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_transcription(base_url, audio_file):
    """Test transcription endpoint"""
    print(f"\n=== Testing Transcription ===")
    print(f"Audio file: {audio_file}")
    
    try:
        with open(audio_file, 'rb') as f:
            files = {'file': (audio_file, f, 'audio/wav')}
            
            start_time = time.time()
            response = requests.post(f"{base_url}/transcribe", files=files)
            end_time = time.time()
            
            print(f"Status Code: {response.status_code}")
            print(f"Response Time: {end_time - start_time:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Transcription: {result.get('transcription', 'N/A')}")
                print(f"Duration: {result.get('duration', 'N/A')}s")
                print(f"Sample Rate: {result.get('sample_rate', 'N/A')}Hz")
                return True
            else:
                print(f"Error: {response.text}")
                return False
                
    except Exception as e:
        print(f"Transcription test failed: {e}")
        return False

def test_error_handling(base_url):
    """Test error handling"""
    print(f"\n=== Testing Error Handling ===")
    
    # Test with no file
    try:
        response = requests.post(f"{base_url}/transcribe")
        print(f"No file test: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"No file test error: {e}")
    
    # Test with invalid file type
    try:
        with tempfile.NamedTemporaryFile(suffix='.txt', mode='w') as f:
            f.write("This is not an audio file")
            f.flush()
            
            with open(f.name, 'rb') as tf:
                files = {'file': ('test.txt', tf, 'text/plain')}
                response = requests.post(f"{base_url}/transcribe", files=files)
                print(f"Invalid file type test: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Invalid file type test error: {e}")
    
    # Test with audio that's too short
    try:
        short_audio = generate_test_audio(duration=2, filename="short_test.wav")
        with open(short_audio, 'rb') as f:
            files = {'file': (short_audio, f, 'audio/wav')}
            response = requests.post(f"{base_url}/transcribe", files=files)
            print(f"Short audio test: {response.status_code} - {response.json()}")
        os.unlink(short_audio)
    except Exception as e:
        print(f"Short audio test error: {e}")

def test_concurrent_requests(base_url, audio_file, num_requests=5):
    """Test concurrent requests"""
    print(f"\n=== Testing Concurrent Requests ===")
    print(f"Sending {num_requests} concurrent requests...")
    
    import concurrent.futures
    import threading
    
    results = []
    
    def send_request():
        try:
            with open(audio_file, 'rb') as f:
                files = {'file': (audio_file, f, 'audio/wav')}
                start_time = time.time()
                response = requests.post(f"{base_url}/transcribe", files=files)
                end_time = time.time()
                
                return {
                    'status_code': response.status_code,
                    'response_time': end_time - start_time,
                    'success': response.status_code == 200
                }
        except Exception as e:
            return {'error': str(e), 'success': False}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(send_request) for _ in range(num_requests)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    # Analyze results
    successful = sum(1 for r in results if r.get('success', False))
    avg_time = np.mean([r['response_time'] for r in results if 'response_time' in r])
    
    print(f"Successful requests: {successful}/{num_requests}")
    print(f"Average response time: {avg_time:.2f}s")
    
    return successful == num_requests

def main():
    parser = argparse.ArgumentParser(description="Test NeMo ASR API")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--audio-file", help="Audio file to test with")
    parser.add_argument("--generate-audio", action="store_true", help="Generate test audio")
    parser.add_argument("--duration", type=int, default=8, help="Duration of generated audio")
    parser.add_argument("--concurrent-tests", type=int, default=3, help="Number of concurrent requests")
    parser.add_argument("--skip-health", action="store_true", help="Skip health check tests")
    parser.add_argument("--skip-errors", action="store_true", help="Skip error handling tests")
    parser.add_argument("--skip-concurrent", action="store_true", help="Skip concurrent tests")
    
    args = parser.parse_args()
    
    print(f"Testing API at: {args.base_url}")
    
    # Generate or use provided audio file
    if args.generate_audio or not args.audio_file:
        audio_file = generate_test_audio(duration=args.duration)
    else:
        audio_file = args.audio_file
        if not os.path.exists(audio_file):
            print(f"Audio file not found: {audio_file}")
            return
    
    try:
        all_tests_passed = True
        
        # Health check test
        if not args.skip_health:
            if not test_health_check(args.base_url):
                print("❌ Health check test failed")
                all_tests_passed = False
            else:
                print("✅ Health check test passed")
        
        # Transcription test
        if not test_transcription(args.base_url, audio_file):
            print("❌ Transcription test failed")
            all_tests_passed = False
        else:
            print("✅ Transcription test passed")
        
        # Error handling tests
        if not args.skip_errors:
            test_error_handling(args.base_url)
            print("✅ Error handling tests completed")
        
        # Concurrent request tests
        if not args.skip_concurrent:
            if not test_concurrent_requests(args.base_url, audio_file, args.concurrent_tests):
                print("❌ Concurrent requests test failed")
                all_tests_passed = False
            else:
                print("✅ Concurrent requests test passed")
        
        # Overall result
        print(f"\n{'='*50}")
        if all_tests_passed:
            print("🎉 All tests passed!")
        else:
            print("❌ Some tests failed. Check the logs above.")
        
    finally:
        # Cleanup generated audio file
        if args.generate_audio or not args.audio_file:
            try:
                os.unlink(audio_file)
                print(f"Cleaned up test audio: {audio_file}")
            except:
                pass

if __name__ == "__main__":
    main()
