import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";
import Index from "./pages/Index";
import ConfigPage from "./pages/ConfigPage";
import ReferencesPage from "./pages/ReferencesPage";
import ModalityPage from "./pages/ModalityPage";
import ImportPage from "./pages/ImportPage";
import AnnotatePage from "./pages/AnnotatePage";
import PipelinePage from "./pages/PipelinePage";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <SidebarProvider>
          <div className="min-h-screen flex w-full">
            <AppSidebar />
            <div className="flex-1 flex flex-col min-w-0">
              <header className="h-12 flex items-center border-b bg-background/80 backdrop-blur-sm sticky top-0 z-10">
                <SidebarTrigger className="ml-3" />
                <span className="ml-3 text-xs text-muted-foreground font-heading">
                  SyncScope — Multimodal Interpersonal Synchrony Analyzer
                </span>
              </header>
              <main className="flex-1 overflow-auto">
                <Routes>
                  <Route path="/" element={<Index />} />
                  <Route path="/import" element={<ImportPage />} />
                  <Route path="/annotate" element={<AnnotatePage />} />
                  <Route path="/pipeline" element={<PipelinePage />} />
                  <Route path="/config" element={<ConfigPage />} />
                  <Route path="/references" element={<ReferencesPage />} />
                  <Route path="/modality/:modality" element={<ModalityPage />} />
                  <Route path="*" element={<NotFound />} />
                </Routes>
              </main>
            </div>
          </div>
        </SidebarProvider>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
