import time
import os
from statistics import mean, median
from serpapi import GoogleSearch
from perplexity import Perplexity

# Configuration
SERPAPI_KEY = os.getenv('SERPAPI_API_KEY', 'your_serpapi_key_here')
PERPLEXITY_KEY = os.getenv('PERPLEXITY_API_KEY', 'your_perplexity_key_here')

# Test queries
TEST_QUERIES = [
    "Python programming best practices",
    "machine learning algorithms 2025",
    "climate change statistics",
    "artificial intelligence trends",
    "stock market analysis tools"
]

def measure_serpapi_latency(query):
    """Measure latency for SERP API with Google engine"""
    try:
        start_time = time.time()
        
        search = GoogleSearch({
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": 10  # Number of results
        })
        
        results = search.get_dict()
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        
        return {
            'success': True,
            'latency_ms': latency,
            'results_count': len(results.get('organic_results', []))
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'latency_ms': None
        }

def measure_perplexity_latency(query):
    """Measure latency for Perplexity Search API"""
    try:
        start_time = time.time()
        
        client = Perplexity(api_key=PERPLEXITY_KEY)
        search = client.search.create(
            query=query,
            max_results=10,
            max_tokens_per_page=1024
        )
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        
        return {
            'success': True,
            'latency_ms': latency,
            'results_count': len(search.results)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'latency_ms': None
        }

def run_latency_comparison(queries, iterations=3):
    """Run latency comparison across multiple queries and iterations"""
    results = {
        'serpapi': {'latencies': [], 'errors': 0},
        'perplexity': {'latencies': [], 'errors': 0}
    }
    
    print(f"Testing {len(queries)} queries with {iterations} iterations each...\n")
    
    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}/{len(queries)}]: {query}")
        print("-" * 80)
        
        # Test SERP API
        serpapi_latencies = []
        for iteration in range(iterations):
            result = measure_serpapi_latency(query)
            if result['success']:
                serpapi_latencies.append(result['latency_ms'])
                print(f"  SERP API (iter {iteration+1}): {result['latency_ms']:.2f} ms | Results: {result['results_count']}")
            else:
                results['serpapi']['errors'] += 1
                print(f"  SERP API (iter {iteration+1}): ERROR - {result['error']}")
            
            time.sleep(0.5)  # Small delay between requests
        
        # Test Perplexity API
        perplexity_latencies = []
        for iteration in range(iterations):
            result = measure_perplexity_latency(query)
            if result['success']:
                perplexity_latencies.append(result['latency_ms'])
                print(f"  Perplexity (iter {iteration+1}): {result['latency_ms']:.2f} ms | Results: {result['results_count']}")
            else:
                results['perplexity']['errors'] += 1
                print(f"  Perplexity (iter {iteration+1}): ERROR - {result['error']}")
            
            time.sleep(0.5)  # Small delay between requests
        
        # Store latencies
        results['serpapi']['latencies'].extend(serpapi_latencies)
        results['perplexity']['latencies'].extend(perplexity_latencies)
        
        # Display query summary
        if serpapi_latencies and perplexity_latencies:
            avg_serp = mean(serpapi_latencies)
            avg_perp = mean(perplexity_latencies)
            faster = "SERP API" if avg_serp < avg_perp else "Perplexity"
            diff = abs(avg_serp - avg_perp)
            print(f"\n  Query Summary: {faster} was faster by {diff:.2f} ms")
    
    return results

def display_final_results(results):
    """Display comprehensive comparison results"""
    print("\n" + "=" * 80)
    print("FINAL COMPARISON RESULTS")
    print("=" * 80)
    
    for api_name, data in results.items():
        print(f"\n{api_name.upper()}:")
        if data['latencies']:
            print(f"  Total Requests: {len(data['latencies'])}")
            print(f"  Successful: {len(data['latencies'])}")
            print(f"  Failed: {data['errors']}")
            print(f"  Average Latency: {mean(data['latencies']):.2f} ms")
            print(f"  Median Latency: {median(data['latencies']):.2f} ms")
            print(f"  Min Latency: {min(data['latencies']):.2f} ms")
            print(f"  Max Latency: {max(data['latencies']):.2f} ms")
        else:
            print(f"  No successful requests")
            print(f"  Failed: {data['errors']}")
    
    # Overall comparison
    if results['serpapi']['latencies'] and results['perplexity']['latencies']:
        serp_avg = mean(results['serpapi']['latencies'])
        perp_avg = mean(results['perplexity']['latencies'])
        
        print(f"\n" + "=" * 80)
        print("WINNER:")
        if serp_avg < perp_avg:
            diff_percent = ((perp_avg - serp_avg) / perp_avg) * 100
            print(f"  SERP API is faster by {perp_avg - serp_avg:.2f} ms ({diff_percent:.1f}% faster)")
        else:
            diff_percent = ((serp_avg - perp_avg) / serp_avg) * 100
            print(f"  Perplexity is faster by {serp_avg - perp_avg:.2f} ms ({diff_percent:.1f}% faster)")
        print("=" * 80)

if __name__ == "__main__":
    # Run the comparison
    results = run_latency_comparison(TEST_QUERIES, iterations=3)
    
    # Display final results
    display_final_results(results)
