from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
#
## ROUTERS
#from app.routes.estudiante_routes import router as estudiante_router 
#from app.routes.consulta_routes import router as consulta_router
#from app.routes.clinica_routes import router as clinica_router
#from app.routes.signos_vitales_routes import router as signos_vitales_router
#from app.routes.derivacion_routes import router as derivacion_router
#from app.routes.emergencia_routes import router as emergencia_router
#from app.routes.rol_routes import router as rol_router
#from app.routes.usuario_routes import router as usuario_router
#from app.routes.facultad_routes import router as facultad_router
#from app.routes.programa_routes import router as programa_router
#from app.routes.auth_routes import router as auth_router
from routes.rag_routes import router as rag_router

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5173",
    "https://upgraded-space-lamp-g459wvwjvw6639vvj-5173.app.github.dev",
    "https://fronted-ten-iota.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # o ["*"] para permitir todos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REGISTRO DE ROUTERS

#app.include_router(estudiante_router)
#app.include_router(consulta_router)
#app.include_router(clinica_router)
#app.include_router(signos_vitales_router)
#app.include_router(derivacion_router)
#app.include_router(emergencia_router)
#app.include_router(rol_router)
#app.include_router(usuario_router)
#app.include_router(facultad_router)
#app.include_router(programa_router)
#app.include_router(auth_router)
app.include_router(rag_router)







