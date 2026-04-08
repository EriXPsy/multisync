import {
  Brain,
  Eye,
  Heart,
  Users,
  BarChart3,
  BookOpen,
  Settings2,
  Activity,
  Upload,
  Video,
  GitBranch,
} from "lucide-react";
import { NavLink } from "@/components/NavLink";
import { useLocation } from "react-router-dom";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
  SidebarFooter,
  useSidebar,
} from "@/components/ui/sidebar";

const mainItems = [
  { title: "Unified Timeline", url: "/", icon: BarChart3 },
  { title: "Import Data", url: "/import", icon: Upload },
  { title: "Video Annotator", url: "/annotate", icon: Video },
  { title: "Analysis Pipeline", url: "/pipeline", icon: GitBranch },
  { title: "Configuration", url: "/config", icon: Settings2 },
  { title: "References", url: "/references", icon: BookOpen },
];

const modalityItems = [
  { title: "Neural", url: "/modality/neural", icon: Brain, className: "text-neural" },
  { title: "Behavioral", url: "/modality/behavioral", icon: Eye, className: "text-behavioral" },
  { title: "Biosynchrony", url: "/modality/bio", icon: Heart, className: "text-bio" },
  { title: "Psycho-synchrony", url: "/modality/psycho", icon: Users, className: "text-psycho" },
];

export function AppSidebar() {
  const { state } = useSidebar();
  const collapsed = state === "collapsed";
  const location = useLocation();
  const currentPath = location.pathname;

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader className="p-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-sidebar-primary/20 flex items-center justify-center">
            <Activity className="w-4 h-4 text-sidebar-primary" />
          </div>
          {!collapsed && (
            <div className="animate-slide-in">
              <h1 className="font-heading text-sm font-bold text-sidebar-primary">SyncScope</h1>
              <p className="text-[10px] text-sidebar-foreground/60">Multimodal Synchrony</p>
            </div>
          )}
        </div>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel className="text-sidebar-foreground/40 text-[10px] uppercase tracking-wider">
            Analysis
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {mainItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild>
                    <NavLink
                      to={item.url}
                      end
                      className="hover:bg-sidebar-accent/50"
                      activeClassName="bg-sidebar-accent text-sidebar-primary font-medium"
                    >
                      <item.icon className="mr-2 h-4 w-4" />
                      {!collapsed && <span>{item.title}</span>}
                    </NavLink>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel className="text-sidebar-foreground/40 text-[10px] uppercase tracking-wider">
            Modalities
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {modalityItems.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild>
                    <NavLink
                      to={item.url}
                      className="hover:bg-sidebar-accent/50"
                      activeClassName="bg-sidebar-accent text-sidebar-primary font-medium"
                    >
                      <item.icon className={`mr-2 h-4 w-4 ${item.className}`} />
                      {!collapsed && <span>{item.title}</span>}
                    </NavLink>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="p-4">
        {!collapsed && (
          <div className="text-[10px] text-sidebar-foreground/30 space-y-1">
            <p>Grounded in 20+ studies</p>
            <p>Epoch-aggregated alignment</p>
          </div>
        )}
      </SidebarFooter>
    </Sidebar>
  );
}
