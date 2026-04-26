import { useLocation } from "react-router-dom";
import { useEffect } from "react";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error("404 Error: User attempted to access non-existent route:", location.pathname);
  }, [location.pathname]);

  return (
    <div className="site-gutter-x flex min-h-screen-dvh items-center justify-center bg-muted py-8">
      <div className="w-full max-w-md text-center">
        <h1 className="mb-3 text-[length:clamp(2rem,calc(1.2rem+2.5vw),2.75rem)] font-bold sm:mb-4">
          404
        </h1>
        <p className="mb-4 text-[length:clamp(1rem,calc(0.88rem+0.45vw),1.3rem)] text-muted-foreground sm:text-xl">
          Oops! Page not found
        </p>
        <a
          href="/"
          className="text-primary underline underline-offset-4 min-h-11 inline-flex items-center justify-center rounded-md px-2 py-1.5 hover:text-primary/90"
        >
          Return to Home
        </a>
      </div>
    </div>
  );
};

export default NotFound;
