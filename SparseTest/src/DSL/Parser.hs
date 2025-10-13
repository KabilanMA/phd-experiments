{-# LANGUAGE OverloadedStrings #-}

module DSL.Parser (parseEinsum, parseExpr, parseIndex, parseFile) where

import DSL.AST
import Control.Monad (void)
import Data.Void (Void)
import Text.Megaparsec
import Text.Megaparsec.Char
import qualified Text.Megaparsec.Char.Lexer as L

type Parser = Parsec Void String

-- | Lexer
sc :: Parser ()
sc = L.space space1 (L.skipLineComment "#") empty

lexeme :: Parser a -> Parser a
lexeme = L.lexeme sc

symbol :: String -> Parser String
symbol = L.symbol sc

-- | Parse an identifier (e.g., tensor name)
identifier :: Parser String
identifier = lexeme ((:) <$> letterChar <*> many alphaNumChar)

-- | Parse an index (single character)
parseIndex :: Parser Index
parseIndex = Ix <$> lowerChar

-- | Parse list of indices like [i,j,k]
parseIndexList :: Parser [Index]
parseIndexList = between (symbol "[") (symbol "]") (parseIndex `sepBy1` symbol ",")

-- | Parse constants, variables, and parenthesized exprs
parseAtom :: Parser Expr
parseAtom =
      try parseVar
  <|> try parseConst
  <|> between (symbol "(") (symbol ")") parseExpr

parseConst :: Parser Expr
parseConst = EConstDouble <$> lexeme L.float

parseVar :: Parser Expr
parseVar = do
  name <- identifier
  idxs <- optional parseIndexList
  return $ EVar name (maybe [] id idxs)

-- | Parse unary expressions like -A[i]
parseUnary :: Parser Expr
parseUnary =
      (EUnary UNegate <$> (symbol "-" *> parseUnary))
  <|> parseAtom

-- | Parse binary operations with precedence
parseMulDiv :: Parser Expr
parseMulDiv = makeBinOps parseUnary [("*", BMul), ("/", BDiv)]

parseAddSub :: Parser Expr
parseAddSub = makeBinOps parseMulDiv [("+", BAdd), ("-", BSub)]

parseExpr :: Parser Expr
parseExpr = parseAddSub

-- | Generic binary operator helper
makeBinOps :: Parser Expr -> [(String, BinaryOp)] -> Parser Expr
makeBinOps lower ops = foldl makeParser lower ops
  where
    makeParser p (sym, bop) =
      do
        let op = symbol sym >> return (EBinary bop)
        chainl1 p op

-- | Chain left helper (Megaparsec doesn't export one)
chainl1 :: Parser a -> Parser (a -> a -> a) -> Parser a
chainl1 p op = do
  x <- p
  rest x
  where
    rest x = (do
      f <- op
      y <- p
      rest (f x y)) <|> return x

-- | Parse einsum statement like: C[i] = A[i,j] * B[j,i]
parseEinsum :: Parser EinsumStmt
parseEinsum = do
  out <- identifier
  idxs <- parseIndexList
  void $ symbol "="
  e <- parseExpr
  return $ EinsumStmt
    { outName = out
    , outIdxs = idxs
    , inputs = []          -- we’ll fill this in later if you want decl parsing
    , expr = e
    , annots = []
    }

-- | Parse a full file with multiple einsum statements
parseFile :: Parser [EinsumStmt]
parseFile = many (parseEinsum <* optional eol)

