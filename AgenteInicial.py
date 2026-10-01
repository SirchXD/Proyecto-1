import json

import numpy as np
import pandas as pd


class AgenteUtilidadJuridico:
    """Agente basado en utilidad para la evaluación de evidencia procesal penal,
    generación de propuesta jurídica trazable y benchmarking contra sentencias reales.
    """

    def __init__(self, cnije_csv_path=None, sentencia_json_path=None):
        self.cnije_path = cnije_csv_path
        self.sentencia_path = sentencia_json_path
        self.evidencias = []
        self.media_pena_cnije = 5.0  # Valor histórico por defecto (años)

    def cargar_datos_cnije(self, df_cnije=None):
        """Carga y procesa datos abiertos del CNIJE (INEGI) - Materia Penal.

        Estructura típica: [id_causa, delito_principal, tipo_resolucion, pena_anios,
        medida_cautelar]
        """
        if df_cnije is not None:
            self.df_cnije = df_cnije
        else:
            data = {
                "id_causa": ["C-001", "C-002", "C-003", "C-004", "C-005"],
                "delito_principal": [
                    "Robo Calificado",
                    "Robo Calificado",
                    "Robo Calificado",
                    "Homicidio",
                    "Homicidio",
                ],
                "tipo_resolucion": [
                    "Condenatoria",
                    "Condenatoria",
                    "Absolutoria",
                    "Condenatoria",
                    "Condenatoria",
                ],
                "pena_anios": [4.5, 6.0, 0.0, 15.0, 18.0],
                "medida_cautelar": [
                    "Prisión Preventiva",
                    "Prisión Preventiva",
                    "Libertad",
                    "Prisión Preventiva",
                    "Prisión Preventiva",
                ],
            }
            self.df_cnije = pd.DataFrame(data)

        delito_target = "Robo Calificado"
        filtro = (
            (self.df_cnije["delito_principal"] == delito_target)
            & (self.df_cnije["tipo_resolucion"] == "Condenatoria")
        )
        if not self.df_cnije[filtro].empty:
            self.media_pena_cnije = self.df_cnije[filtro]["pena_anios"].mean()

    def extraer_evidencia_expediente(self, expediente_data=None):
        """Extrae y parametriza las pruebas desahogadas en la versión pública del expediente."""
        if expediente_data is not None:
            self.evidencias = expediente_data.get("evidencias", [])
            self.delito_imputado = expediente_data.get("delito", "Robo Calificado")
        else:
            self.delito_imputado = "Robo Calificado"
            self.evidencias = [
                {
                    "id": "E1",
                    "tipo": "Pericial Dactiloscópica",
                    "peso_probatorio": 0.90,
                    "impugnada": False,
                    "fundamento": "Art. 272 CNPP",
                },
                {
                    "id": "E2",
                    "tipo": "Testimonial de Oídas",
                    "peso_probatorio": 0.45,
                    "impugnada": True,
                    "fundamento": "Art. 359 CNPP",
                },
                {
                    "id": "E3",
                    "tipo": "Informe Policial Homologado (IPH)",
                    "peso_probatorio": 0.80,
                    "impugnada": False,
                    "fundamento": "Art. 132 CNPP",
                },
                {
                    "id": "E4",
                    "tipo": "Cámara de CCTV Privada",
                    "peso_probatorio": 0.95,
                    "impugnada": False,
                    "fundamento": "Art. 217 CNPP",
                },
            ]

    def funcion_utilidad(self, alternativa):
        """Evaluación multi-atributo: U(s) = w1*SustentoProbatorio + w2*CoherenciaNormativa
        + w3*ProporcionalidadCNIJE - w4*RiesgoProcesal
        """
        pesos_efectivos = [
            e["peso_probatorio"] * (0.3 if e["impugnada"] else 1.0)
            for e in self.evidencias
        ]
        sustento = np.mean(pesos_efectivos) if pesos_efectivos else 0.0

        coherencia = 1.0 if alternativa["calificacion_juridica_valida"] else 0.0

        desviacion_media = abs(alternativa["pena_propuesta"] - self.media_pena_cnije)
        proporcionalidad = max(0.0, 1.0 - (desviacion_media / 10.0))

        riesgo = (
            0.4
            if alternativa["medida_cautelar"] == "Prisión Preventiva"
            and sustento < 0.6
            else 0.0
        )

        w1, w2, w3, w4 = 0.40, 0.30, 0.20, 0.10
        utilidad = (
            (w1 * sustento)
            + (w2 * coherencia)
            + (w3 * proporcionalidad)
            - (w4 * riesgo)
        )

        return {
            "utilidad_total": round(utilidad, 4),
            "desglose": {
                "sustento_probatorio": round(sustento, 4),
                "coherencia_normativa": round(coherencia, 4),
                "proporcionalidad_cnije": round(proporcionalidad, 4),
                "riesgo_procesal": round(riesgo, 4),
            },
        }

    def generar_propuesta_optima(self):
        """Explora el espacio de decisiones y retorna la alternativa con máxima utilidad U(s)."""
        espacio_alternativas = [
            {
                "id_propuesta": "P1_Condena_Alta",
                "calificacion_juridica_valida": True,
                "pena_propuesta": 7.0,
                "medida_cautelar": "Prisión Preventiva",
                "fundamento": "Robo Calificado con Agravante Máxima",
            },
            {
                "id_propuesta": "P2_Condena_Ajustada_CNIJE",
                "calificacion_juridica_valida": True,
                "pena_propuesta": 5.2,
                "medida_cautelar": "Prisión Preventiva",
                "fundamento": "Robo Calificado (Alineado a Media CNIJE-INEGI)",
            },
            {
                "id_propuesta": "P3_Absolutoria_Duda_Razonable",
                "calificacion_juridica_valida": False,
                "pena_propuesta": 0.0,
                "medida_cautelar": "Libertad",
                "fundamento": "Insuficiencia Probatoria",
            },
        ]

        resultados = []
        for alt in espacio_alternativas:
            evaluacion = self.funcion_utilidad(alt)
            alt_evaluada = {**alt, **evaluacion}
            resultados.append(alt_evaluada)

        optima = max(resultados, key=lambda x: x["utilidad_total"])
        return optima, resultados

    def benchmark_vs_sentencia_real(self, propuesta_optima, sentencia_real):
        """Compara la postura del agente contra la resolución judicial real."""
        coincidencia_sentido = (
            propuesta_optima["pena_propuesta"] > 0
        ) == sentencia_real["condenatoria"]
        error_pena = abs(
            propuesta_optima["pena_propuesta"] - sentencia_real["pena_dictaminada"]
        )
        mapa_trazabilidad = [
            (
                f"Prueba {e['id']} ({e['tipo']}) | Peso: {e['peso_probatorio']} | "
                f"Fundamento: {e['fundamento']}"
            )
            for e in self.evidencias
            if not e["impugnada"]
        ]

        return {
            "coincidencia_sentido": coincidencia_sentido,
            "diferencia_pena_anios": round(error_pena, 2),
            "utilidad_alcanzada": propuesta_optima["utilidad_total"],
            "sentencia_real_pena": sentencia_real["pena_dictaminada"],
            "propuesta_agente_pena": propuesta_optima["pena_propuesta"],
            "mapa_trazabilidad": mapa_trazabilidad,
        }


if __name__ == "__main__":
    agente = AgenteUtilidadJuridico()
    agente.cargar_datos_cnije()
    agente.extraer_evidencia_expediente()
    optima, todas = agente.generar_propuesta_optima()
    sentencia_real = {"condenatoria": True, "pena_dictaminada": 5.0}
    benchmark = agente.benchmark_vs_sentencia_real(optima, sentencia_real)

    print("=== MÁXIMA UTILIDAD ALCANZADA POR EL AGENTE ===")
    print(json.dumps(optima, indent=2, ensure_ascii=False))
    print("\n=== COMPARATIVA BENCHMARK VS. SENTENCIA JUDICIAL REAL ===")
    print(json.dumps(benchmark, indent=2, ensure_ascii=False))
