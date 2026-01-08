# Rethinking RAG: Pipelines Are the Past, Agentic Is the Future

[RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) is essential. It's also hard to get right. Production RAG systems often disappoint, and the response is typically to pile on workarounds: [HyDE](https://docs.haystack.deepset.ai/docs/hypothetical-document-embeddings-hyde), reranking, query expansion, chunk overlap tuning. Complexity grows, while results don't necessarily improve.

The problem isn't the workarounds. It's what they're working around.

Legacy RAG frameworks such as LangChain are built on an obsolete model that dates to 2023, before LLM tool calling changed everything. They use static pipelines when what's needed is fundamentally different: **letting the LLM reason about retrieval rather than blindly executing a predetermined flow**.

Recent research is clear. The future of RAG is agentic.

## The Past: The Pipeline

Traditional RAG follows a rigid pipeline: query → retrieve → generate. 

### LangChain (Python): The OG Pipeline

As I've previously noted, LangChain owes its prominence not to its quality, but to being the first mover. In this case, this has become a major liability, as its RAG approach is obsolete.

Here's how LangChain implements RAG using [LCEL (LangChain Expression Language)](https://python.langchain.com/docs/concepts/lcel/):

```python
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# The canonical RAG pipeline
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
) 

# Or with RunnableParallel for slightly more control
rag_chain = (
    RunnableParallel(context=retriever | format_docs, question=RunnablePassthrough())
    | qa_prompt 
    | llm
)
```

This pattern appears across tutorials, documentation, and production systems. A retriever runs in response to a query and the results are added to the prompt before the LLM generates an answer. Welcome to 2023.

### LangChain4j: Difference Language, Same Old Model

As you would expect from its name, [LangChain4j](https://docs.langchain4j.dev/tutorials/rag/) brings the same model to the JVM.
The tutorial explicitly defines RAG in this limited, obsolete way:
"Simply put, RAG is the way to find and inject relevant pieces of information from your data into the prompt before sending it to the LLM."

Here's example code from the [Easy RAG example](https://github.com/langchain4j/langchain4j-examples/blob/main/rag-examples/src/main/java/_1_easy/Easy_RAG_Example.java):

```java
public class Easy_RAG_Example {

    public static void main(String[] args) {
        List<Document> documents = loadDocuments(toPath("documents/"), glob("*.txt"));

        Assistant assistant = AiServices.builder(Assistant.class)
            .chatModel(CHAT_MODEL)
            .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
            .contentRetriever(createContentRetriever(documents))
            .build();

        startConversationWith(assistant);
    }

    private static ContentRetriever createContentRetriever(List<Document> documents) {
        InMemoryEmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        EmbeddingStoreIngestor.ingest(documents, embeddingStore);
        return EmbeddingStoreContentRetriever.from(embeddingStore);
    }
}
```

The `ContentRetriever` supplements the prompt to help answer queries. The [documentation explicitly describes](https://docs.langchain.com/oss/python/langchain/rag) this as a "two-step chain" that provides "reduced latency at the expense of flexibility."

Spring AI uses a [similar](https://docs.spring.io/spring-ai/reference/api/retrieval-augmented-generation.html) (if more flexible) pipeline model.

### What's Wrong With Pipelines?

You can build more elaborate pipelines with these old school RAG frameworks, but can't  fix the core problem with the approach.
The pipeline model has fundamental problems that no amount of tuning can fix:

**1. Static Retrieval**
The retriever retrieves once and hopes for the best. If the initial query doesn't match how documents are indexed, you get poor results. Even if hybrid search is available, the balance between vector and full-text search will be fixed for all queries. The LLM never gets a chance to try a different, highly contextual approach.

**2. No Self-Correction**
If retrieved chunks don't actually answer the question, the pipeline has no mechanism to recognize this and try again. It has to trust whatever comes back from the vector store. Nor can it decide how hard to try in a particular scenario: Is this a legal matter or a casual question? Should it dig deep, regardless of latency, or give up quickly?

**3. Context Blindness**
The retriever may not know what the LLM learned from previous turns in a conversation. Each retrieval is isolated, unable to build on prior context.

**4. Workaround Proliferation**
To compensate for these limitations, teams bolt on increasingly complex preprocessing: HyDE (Hypothetical Document Embeddings) to bridge the query-document vocabulary gap, rerankers to fix retrieval ordering, query expansion to catch more results, overlap tuning to hope chunk boundaries don't split relevant content.

> Chunk boundaries are a particular risk. No chunk size can be optimal for all documents. Retrieved chunks are often split in the wrong place, making them misleading.

### The Research Agrees

Recent research is unambiguous:

- [**Anthropic**](https://www.anthropic.com/engineering/multi-agent-research-system): *"Traditional approaches using RAG use static retrieval... You can't hardcode a fixed path for exploring complex topics, as the process is inherently dynamic and path-dependent."* Their multi-agent research system **outperformed single-agent approaches by 90.2%**.
- [**NVIDIA**](https://developer.nvidia.com/blog/traditional-rag-vs-agentic-rag-why-ai-agents-need-dynamic-knowledge-to-get-smarter/): Agentic RAG *"refines queries using reasoning, turning RAG into a sophisticated tool"* versus traditional RAG's "lack of reasoning" and "context blindness."
- [**arXiv survey**](https://arxiv.org/abs/2501.09136) (January 2025): *"Traditional RAG systems are constrained by static workflows and lack the adaptability required for multistep reasoning and complex task management."*
- **Comparative studies** ([TechRxiv](https://www.techrxiv.org/users/876974/articles/1325941-traditional-rag-vs-agentic-rag-a-comparative-study-of-retrieval-augmented-systems), [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5188363)): 80% improvement in retrieval quality and 90% of users preferring agentic systems.

Pipeline RAG is not state of the art. It's a first-generation approach that is now largely obsolete.

## The Future: Agentic RAG

What if we let the LLM control the retrieval process? Instead of a rigid pipeline, give the model tools to search, evaluate results, refine queries, and iterate until it finds what it needs.

This is **agentic** RAG. The LLM becomes an active participant in retrieval rather than a passive consumer of whatever the pipeline produces.

## Embabel: Agentic RAG Done Right

Embabel was built from the ground up for agentic AI. Our new RAG support sits cleanly on top of the core framework through our `LlmReference` abstraction.

### ToolishRag: Fine-Grained Search as LLM Tools

`ToolishRag` wraps any `SearchOperations` implementation and exposes its capabilities as individual tools the LLM can invoke.

`ToolishRag` inspects what interfaces the underlying store implements and only exposes tools for operations the store actually supports.

For example, A Lucene store gets vector search, text search, regex search, and result expansion tools. A Neo database gets vector and full text and tools to navigate source document structure from chunks. A simple vector database adapter gets only vector search. The LLM will always be equipped to with the best possible toolkit for the store in use, without the developer needing to write any extra code.

The basic tools are vector and text search, but chunk navigation tools are also extremely important.

Some key tools:

**Vector Search** — Semantic similarity search with configurable top-K and threshold:
```kotlin
@LlmTool(description = "Perform vector search. Specify topK and similarity threshold from 0-1")
fun vectorSearch(query: String, topK: Int, threshold: ZeroToOne): String
```

**Text Search** — BM25 search with full Lucene syntax support:
```kotlin
@LlmTool(description = "Perform BM25 search with Lucene syntax: +term, -term, \"phrases\", wildcards (*), fuzzy (~)")
fun textSearch(query: String, topK: Int, threshold: ZeroToOne): String
```

**Chunk Expansion** — Broaden context around a retrieved chunk:
```kotlin
@LlmTool(description = "Given a chunk ID, expand to surrounding chunks")
fun broadenChunk(chunkId: String, chunksToAdd: Int = 2): String
```
This is extremely important as it mitigates the problem of
    chunk boundary splitting relevant content. If the LLM seems the start or end of what appears to be a promising reef of content, it can continue mining.

Similarly, for document-structured stores such as Neo4j:

**Zoom Out** — Navigate to parent sections for broader context:
```kotlin
@LlmTool(description = "Given a content element ID, expand to parent section")
fun zoomOut(id: String): String
```

The LLM decides which tools to use, in what order, with what parameters. It can try a vector search, evaluate the results, decide they're not quite right, and try a text search with different terms. It can find a relevant chunk, then broaden it to see surrounding context. It can zoom out to understand where a chunk fits in the document structure.

**This is fundamentally different from a pipeline.** The LLM is able to reason about retrieval to help achieve its overall goal, not passively receiving predetermined results.

### Using ToolishRag in Practice

The API is simple, consistent and elegant. Rag tools can be added to any LLM interaction via the Embabel `PromptRunner`.
Here's a complete example from a production chatbot:

```java
@EmbabelComponent
public class ChatActions {

    private final ToolishRag toolishRag;
    private final RagbotProperties properties;

    public ChatActions(SearchOperations searchOperations, RagbotProperties properties) {
        this.toolishRag = new ToolishRag(
                "sources",
                "The music criticism written by Robert Schumann: His own writings",
                searchOperations)
                .withHint(TryHyDE.usingConversationContext());
        this.properties = properties;
    }

    @Action(canRerun = true, trigger = UserMessage.class)
    void respond(Conversation conversation, ActionContext context) {
        var assistantMessage = context
                .ai()
                .withLlm(properties.chatLlm())
                .withReference(toolishRag)
                .withTemplate("ragbot")
                .respondWithSystemPrompt(conversation, Map.of(
                        "properties", properties,
                        "voice", properties.voice(),
                        "objective", properties.objective()
                ));
        context.sendMessage(conversation.addMessage(assistantMessage));
    }
}
```

1. `SearchOperations` is injected—it could be Lucene, a Spring AI vector store, Neo4j, or any other implementation
2. `ToolishRag` wraps it and exposes appropriate tools based on the store's capabilities
3. `.withHint(TryHyDE.usingConversationContext())` provides guidance about when to try hypothetical document generation
4. `.withReference(toolishRag)` gives the LLM access to all the search tools

The LLM has full control. It can search multiple times with different queries. It can evaluate whether results are relevant. It can expand context when needed. It can give up gracefully if nothing works—the default goal explicitly says *"Continue search until the question is answered, or you have to give up. Be creative, try different types of queries."* You can customize this part of the prompt.

### Store Agnostic by Design

`ToolishRag` works over any `SearchOperations` implementation. The interface hierarchy uses composition to express capabilities:

```kotlin
interface SearchOperations  // Tag interface
interface VectorSearch : TypeRetrievalOperations
interface TextSearch : TypeRetrievalOperations
interface ResultExpander : SearchOperations
interface RegexSearchOperations : SearchOperations
interface CoreSearchOperations : VectorSearch, TextSearch
```

A store implements what it naturally supports, and  `ToolishRag` adapts. This means you can:

- Use a local Lucene store for development with full search capabilities
- Deploy to production with a managed vector database
- Integrate with Neo4j for graph-enhanced retrieval
- Mix stores for different content types

The LLM sees appropriate tools for whatever store you provide. The abstraction handles the adaptation.

## Why HyDE and Reranking Are Workarounds

Traditional RAG needs HyDE because the query and document vocabularies don't match. HyDE generates a hypothetical answer document, embeds that, and searches for similar real documents. It's clever. It's also a workaround for not being able to iterate on queries.

An agent that can reason about why a query isn't returning good results and reformulate it dynamically addresses the root cause. The agent doesn't need to guess what a good document might look like—it can try different queries and evaluate results.

Traditional RAG needs reranking because initial retrieval is "dumb." You retrieve K documents and hope the relevant ones are in there. A reranker fixes the ordering. It's better than nothing. It's also a workaround for not being able to evaluate retrieval quality.

An agent with the [Corrective RAG pattern](https://arxiv.org/html/2501.09136v3) can judge whether retrieved documents actually answer the question. If quality is low, it retrieves more or reformulates. It doesn't need a separate reranker to fix what shouldn't have been broken in the first place.

Embabel supports HyDE as a *hint*—guidance for the LLM to try hypothetical document generation when semantic search isn't working. But it's not required. It's not baked into a pipeline. It's one option among many that the agent can use as it reasons about the problem.

**Techniques like HyDE and reranking exist to patch fundamental limitations of pipeline architectures. An agent that reasons about retrieval quality, reformulates queries, and iterates until it has good context doesn't need static preprocessing steps—it does the equivalent dynamically and more intelligently.**

## What's Next: Entities in RAG

Chunks are necessary but not sufficient. Real documents have structure: sections, headings, entities, relationships. A chunk that mentions "the CEO" loses meaning without knowing who "the CEO" refers to.

We're extending Embabel's RAG to include entity extraction and graph integration. Entities provide structure above chunks. When the agent retrieves a chunk mentioning "the acquisition," it can traverse to the entity representing that acquisition, find the companies involved, the date, the value—context that pure chunk retrieval would miss.

This is another area where agentic approaches excel. An agent can decide when entity lookup is valuable, when graph traversal helps, when plain text search is enough. A pipeline must choose one approach for all queries.

I'll write more about this soon.

## Conclusion

RAG is essential. Pipeline RAG is inadequate.

Frameworks such as LangChain were built on a pipeline model that treats retrieval as a fixed preprocessing step. This worked for simple demos. It fails for production systems with complex queries, conversational context, and diverse document collections.

Embabel takes a fundamentally different approach. `ToolishRag` exposes fine-grained search operations as tools the LLM controls. The `LlmReference` abstraction integrates RAG into the core agent framework rather than bolting it on as an afterthought. The result is retrieval that adapts, iterates, and reasons—not retrieval that executes a predetermined flow and hopes for the best.

If you're building production RAG systems, you have a choice. You can keep adding workarounds to a fundamentally limited architecture. Or you can adopt an approach that addresses the root cause: letting intelligent agents reason about retrieval rather than executing blind pipelines.

The research is clear. The results are dramatic. Pipeline RAG is the past. Agentic RAG is the future.

Embabel is how you build it.
