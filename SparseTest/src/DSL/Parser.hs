{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}

module DSL.Parser
  ( parseTensorDecls
  , TensorDecl(..)
  , DType(..)
  , LevelFormat(..)
  , Dim(..)
  , TensorFormat
  ) where

import Prelude hiding (takeWhile)
import Data.Void
import Text.Megaparsec
import Text.Megaparsec.Char
import qualified Text.Megaparsec.Char.Lexer as L
import DSL.AST

-- Parser type
type Parser = Parsec Void String

-- -------------------------
-- Lexer helpers
-- -------------------------
sc :: Parser ()
sc = L.space space1 lineComment blockComment
  where
    lineComment  = L.skipLineComment "//"
    blockComment = L.skipBlockComment "/*" "*/"

lexeme :: Parser a -> Parser a
lexeme = L.lexeme sc

symbol :: String -> Parser String
symbol = L.symbol sc

brackets :: Parser a -> Parser a
brackets = between (symbol "[") (symbol "]")

commaSep1 :: Parser a -> Parser [a]
commaSep1 p = p `sepBy1` symbol ","

ident :: Parser String
ident = lexeme ((:) <$> letterChar <*> many (alphaNumChar <|> char '_'))

-- -------------------------
-- Primitive parsers
-- -------------------------
pDType :: Parser DType
pDType = lexeme $ choice
  [ TInt   <$ (string' "int"   <* notFollowedBy alphaNumChar)
  , TFloat <$ (string' "float" <* notFollowedBy alphaNumChar)
  , TDouble<$ (string' "double"<* notFollowedBy alphaNumChar)
  ]

pLevelFormat :: Parser LevelFormat
pLevelFormat = lexeme $ choice
  [ Dense  <$ string "Dense"
  , Sparse <$ string "Sparse"
  ]

pDim :: Parser Dim
pDim = Dim <$> lexeme L.decimal

-- -------------------------
-- Tensor Declaration
-- -------------------------
-- Example:
-- tensor A [3,3] [Dense,Sparse] float
pTensorDecl :: Parser TensorDecl
pTensorDecl = do
  _ <- symbol "tensor"
  name <- ident
  shape <- brackets (pDim `sepBy` symbol ",")
  fmt <- brackets (pLevelFormat `sepBy` symbol ",")
  dtype <- pDType
  return TensorDecl
    { tName   = name
    , tShape  = shape
    , tFormat = fmt
    , tDType  = dtype
    }

-- Parse multiple tensor declarations
parseTensorDecls :: String -> Either (ParseErrorBundle String Void) [TensorDecl]
parseTensorDecls input = parse (sc *> many (pTensorDecl <* sc) <* eof) "<input>" input
