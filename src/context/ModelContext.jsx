import { createContext, useContext, useReducer } from "react";

const initial = {
  availableModels: [],
  selectedModel: null,
  status: "idle", // idle | downloading | loaded | training | error
  progress: 0,
  error: undefined,
  isModelReady: false,
};

function reducer(state, action) {
  switch (action.type) {
    case "SET_MODELS":
      return { ...state, availableModels: action.payload };
    case "SELECT_MODEL":
      return { ...state, selectedModel: action.payload };
    case "STATUS":
      return { ...state, status: action.payload };
    case "PROGRESS":
      return { ...state, progress: action.payload };
    case "ERROR":
      return { ...state, error: action.payload };
    case "READY":
      return { ...state, isModelReady: action.payload };
    default:
      return state;
  }
}

const ModelContext = createContext({ state: initial, dispatch: () => {} });

export function ModelProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initial);
  return (
    <ModelContext.Provider value={{ state, dispatch }}>
      {children}
    </ModelContext.Provider>
  );
}

export const useModel = () => useContext(ModelContext);