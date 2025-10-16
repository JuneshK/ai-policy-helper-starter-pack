import Chat from '../components/Chat';
import AdminPanel from '../components/AdminPanel';

export default function Page() {
  return (
    <main className="min-h-screen  bg-gray-50 text-gray-900 px-6 py-10 font-sans transition-all duration-300">

      {/* Header */}
      <header className="mb-10 text-center">
        <h1 className="text-4xl md:text-5xl font-extrabold text-gray-900 mb-3">
          AI Policy & Product Helper
        </h1>
        <p className="text-gray-600 text-lg md:text-xl">
          Local-first RAG starter. Ingest sample docs, ask questions, and see citations.
        </p>
      </header>

      {/* Admin Panel */}
      <section className="w-full mb-8">
          <AdminPanel />
      </section>

      {/* Chat Panel */}
      <section className="w-full mb-10">
         <div className="flex-1 bg-white p-6 rounded-xl shadow-xl transition-all duration-300">
          <Chat />
          </div>
      </section>

      {/* How to Test */}
      <section className="w-120 bg-white p-6 rounded-2xl shadow-xl transition-all duration-300 mx-auto ">
        <h3 className="text-2xl font-semibold text-gray-900 mb-4">How to Test</h3>
        <ol className="list-decimal list-inside space-y-3 text-gray-700">
          <li>
            Click <b className="text-indigo-600">Ingest Sample Docs</b>.
          </li>
          <li>
            Ask: <i className="text-gray-800">Can a customer return a damaged blender after 20 days?</i>
          </li>
          <li>
            Ask: <i className="text-gray-800">What’s the shipping SLA to East Malaysia for bulky items?</i>
          </li>
        </ol>
      </section>
    </main>
  );
}

